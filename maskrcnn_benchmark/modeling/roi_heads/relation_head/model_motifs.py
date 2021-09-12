# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info
import gc
from collections import Counter
import json
# from .pytorch_DGCNN.main import Classifier
from .DCGNN import DGCNN, GIN
from torch_geometric.data import Data, Batch
import sys
import os
import math
from .dmn import DMNPlus
sys.path.append('%s/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs*self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2] 
        :return: 
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:,:,0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:,:,1].contiguous().view(batch_size, 1, num_obj)

        return joint_prob.view(batch_size, num_obj*num_obj)  @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class DecoderRNN(nn.Module):
    def __init__(self, config, obj_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop):
        super(DecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim

        # obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_type='glove.6B', wv_dim=embed_dim)
        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes)+1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.rnn_drop=rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))
        
        self.init_parameters()

    def init_parameters(self):
        # Use sensible default initializations for parameters.
        with torch.no_grad():
            torch.nn.init.constant_(self.state_linearity.bias, 0.0)
            torch.nn.init.constant_(self.input_linearity.bias, 0.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self, inputs, initial_state=None, labels=None, boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed+1)
            else:
                assert l_batch == 1
                out_dist_sample = F.softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind+1)

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = nms_overlaps(boxes_for_nms).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(torch.cat(out_dists,0), 1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments[0].new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            out_commitments = out_commitments
        else:
            out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments


class LSTMContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(LSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        # obj_embed_vecs = obj_edge_vectors(self.obj_classes,wv_type='glove.6B', wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
                input_size=self.obj_dim+self.embed_dim + 128,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim + 128,
                hidden_dim=self.hidden_dim,
                rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
                input_size=self.embed_dim + self.hidden_dim + self.obj_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_edge,
                dropout=self.dropout_rate if self.nl_edge > 1 else 0,
                bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim+self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.obj_dim))

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep) # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]
        
        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)
        
        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp, #obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
                )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps) # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels, boxes_per_cls, ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = cat((obj_embed2, x, obj_ctx), -1)
            
        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((obj_embed2, x), -1))

        return obj_dists, obj_preds, edge_ctx, None


class PathTransfromerEncoder(nn.Module):

    def __init__(self, config, obj_classes, in_channels,path_length):
        super().__init__()

        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        self.path_length = path_length

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.path_ctx_rnn = nn.TransformerEncoderLayer(d_model=300, nhead=16, dim_feedforward=self.hidden_dim, dropout=0.1)

        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        word_list, pair_dict, position_dict = np.load(f'datasets/vg/VG_paths_{path_length}.npy', allow_pickle=True)

        class2idx = dict(zip(obj_classes, range(self.num_obj_classes)))

        self.idx2paths = {}
        self.idx2position = {}

        for pair, paths in pair_dict.items():
            self.idx2paths[(class2idx[pair[0]], class2idx[pair[1]])] = torch.from_numpy(paths).long()
        
        for pair, paths in position_dict.items():
            self.idx2position[(class2idx[pair[0]], class2idx[pair[1]])] = torch.from_numpy(paths).long()

        word_embed_vecs = obj_edge_vectors(word_list, wv_dir=self.cfg.GLOVE_DIR, wv_type='numberbatch-en', wv_dim=self.embed_dim)
        word_embed_vecs = torch.cat((torch.zeros(1, self.embed_dim), word_embed_vecs))
        self.word_embed = nn.Embedding(len(word_list)+1, self.embed_dim)






    def forward(self, x):
        pass

class PathContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    Path file
    [
        [word list],
        [path list],
        {pair: path index}    
    ]
    pairs =dict=> path_indexs =dict=> path_list =embedding_layer=>  path_embedding
    """
    def __init__(self, config, obj_classes, in_channels,path_length):
        super(PathContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        self.path_length = path_length

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        
        word_list, pair_dict, position_dict = np.load(f'datasets/vg/VG_paths_{path_length}.npy', allow_pickle=True)

        class2idx = dict(zip(obj_classes, range(self.num_obj_classes)))

        self.idx2paths = {}
        self.idx2position = {}

        for pair, paths in pair_dict.items():
            self.idx2paths[(class2idx[pair[0]], class2idx[pair[1]])] = torch.from_numpy(paths).long()
        
        for pair, paths in position_dict.items():
            self.idx2position[(class2idx[pair[0]], class2idx[pair[1]])] = torch.from_numpy(paths).long()

        word_embed_vecs = obj_edge_vectors(word_list, wv_dir=self.cfg.GLOVE_DIR, wv_type='numberbatch-en', wv_dim=self.embed_dim)
        word_embed_vecs = torch.cat((torch.zeros(1, self.embed_dim), word_embed_vecs))
        self.word_embed = nn.Embedding(len(word_list)+1, self.embed_dim)
        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.POSITION_EMBEDDING:
            self.position_embed = nn.Embedding(3, self.embed_dim)

        with torch.no_grad():
            self.word_embed.weight.copy_(word_embed_vecs, non_blocking=True)

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_DIM
        
        # TODO use path parameter
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER

        assert self.nl_obj > 0 

        # neighbor embedding
        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.NEIGHBOR:
            neighbors = np.load(f'datasets/vg/VG_neighbor.npy', allow_pickle=True)

            neighbors = neighbors.item()
            entity_list = neighbors['entities']
            entity_neighbors = neighbors['neighbors']
            neighbors_embed_vecs = obj_edge_vectors(entity_list, wv_dir=self.cfg.GLOVE_DIR, wv_type='numberbatch-en', wv_dim=self.embed_dim)

            self.word2neighbor_embed = nn.Embedding(len(obj_classes), self.embed_dim)

            neigh_embedding = torch.zeros(len(obj_classes), self.embed_dim)

            for e, ns in entity_neighbors.items():
                n_emb = torch.Tensor(len(ns), self.embed_dim)
                for i, n in enumerate(ns):
                    n_emb[i] = torch.mean(neighbors_embed_vecs[n], dim=0, keepdim=False)
                neigh_embedding[class2idx[entity_list[e]]] = torch.mean(n_emb, dim=0, keepdim=False)

            with torch.no_grad():
                self.word2neighbor_embed.weight.copy_(neigh_embedding, non_blocking=True)
            self.neig_line_h = nn.Linear(self.embed_dim*2, self.hidden_dim)



        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.path_ctx_rnn = torch.nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
      
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim

        self.lin_path_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.NEIGHBOR and self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH:
            self.lin_final = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_FUSION=='attention':
            self.visual_projector = nn.Linear(self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.hidden_dim)

    def forward(self, pred_pairs, union_features):
        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH:
            paths_length = []
            paths = []
            positions = []
            empty_path = torch.zeros((1, self.path_length)).long()
            for pair in pred_pairs:
                pair = tuple(pair.tolist())
                if pair in self.idx2paths:
                    paths.append(self.idx2paths[pair])
                    paths_length.append(self.idx2paths[pair].shape[0])
                else:
                    paths.append(empty_path)
                    paths_length.append(1)

            

            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.POSITION_EMBEDDING:
                for pair in pred_pairs:
                    pair = tuple(pair.tolist())
                    if pair in self.idx2position:
                        positions.append(self.idx2position[pair])
                    else:
                        positions.append(empty_path)
            


            all_paths = torch.cat(paths)
            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.POSITION_EMBEDDING:
                all_postions =  torch.cat(positions)

            device = torch.device(self.cfg.MODEL.DEVICE)
            all_paths = all_paths.to(device)
            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.POSITION_EMBEDDING:
                all_postions = all_postions.to(device)

            x = self.word_embed(all_paths)

            

            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.POSITION_EMBEDDING:
                pos_embeding = self.position_embed(all_postions)
                x += pos_embeding

            x = self.path_ctx_rnn(x)[0][:, 0, :]

            x = self.lin_path_h(x)

            path_embedding = x.split(paths_length, dim=0)

            tmp_list = []

            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_FUSION=='attention':
                projector = self.visual_projector(union_features)
                n, _ = projector.shape
                
                query = projector.view(n, -1, 1)
                for q, p in zip(query, path_embedding):
                    attention_scores = p@q

                    attention_scores = torch.softmax(attention_scores.view(1, -1), dim=1)
                    tmp_list.append(attention_scores@p)

            elif self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_FUSION=='max':
                for e in path_embedding:
                    v, _ = torch.max(e, 0, keepdim=True)
                    tmp_list.append(v)
            elif self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_FUSION=='sum':
                tmp_list = [torch.sum(e, 0, keepdim=True) for e in path_embedding]

            path_embedding = tmp_list

            path_embedding = torch.cat(path_embedding)

        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.NEIGHBOR:
            neighbor_embedding = self.word2neighbor_embed(pred_pairs.long())
            m,_,_ = neighbor_embedding.shape
            neighbor_embedding = neighbor_embedding.reshape(m, -1)
            neighbor_embedding = self.neig_line_h(neighbor_embedding)

        if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.NEIGHBOR and self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH:
            # concat, add, dot
            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_NEI_FUSION == "concat":
                total_embedding = torch.cat((path_embedding, neighbor_embedding), dim=1)
                return self.lin_final(total_embedding)

            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_NEI_FUSION == "add":
                return path_embedding + neighbor_embedding
            
            if self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_NEI_FUSION == "dot":
                return path_embedding * neighbor_embedding

        elif self.cfg.MODEL.EXRERNAL_KNOWLEDGE.NEIGHBOR:
            return neighbor_embedding



        return path_embedding   


class TopKNeighborContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    Neighbor file
    [
        [word list],
        {neighbor: top K path}    
    ]
    pairs =dict=> path_indexs =dict=> path_list =embedding_layer=>  path_embedding
    """
    def __init__(self, config, obj_classes, in_channels, path_length):
        super(TopKNeighborContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        self.path_length = path_length

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        
        word_list, neighbor2paths = np.load(f'datasets/vg/VG_top8_paths.npy', allow_pickle=True)


        self.neigh_embedding = torch.zeros(len(obj_classes), 8, path_length)

        word2idx = dict(zip(word_list, range(len(word_list))))

        word_embed_vecs = obj_edge_vectors(word_list, wv_dir=self.cfg.GLOVE_DIR, wv_type='numberbatch-en', wv_dim=self.embed_dim)

        word_embed_vecs = torch.cat((torch.zeros(1, self.embed_dim), word_embed_vecs))
        self.word_embed = nn.Embedding(len(word_list)+1, self.embed_dim)

        with torch.no_grad():
            self.word_embed.weight.copy_(word_embed_vecs, non_blocking=True)


        for i in range(len(obj_classes)):
            if obj_classes[i] == '__background__':
                continue
            for j in range(8):
                length = min(path_length, len(neighbor2paths[obj_classes[i]][j]))
                for k in range(length):
                    w = neighbor2paths[obj_classes[i]][j][k]
                    if w in word2idx: 
                        self.neigh_embedding[i,j,k] = word2idx[w] + 1

        self.neigh_embedding = self.neigh_embedding.long()

        self.hidden_dim = self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_DIM

        self.dmn  = DMNPlus(self.embed_dim, 64,  self.hidden_dim, self.word_embed)




    

    def forward(self, pred_pairs, union_features):

        pred_pairs_shape = pred_pairs.shape

        pred_entity = pred_pairs.view(-1).long()

        paths_fact = self.neigh_embedding[pred_entity]

        x = self.dmn(paths_fact.cuda(), pred_entity.view(-1, 1))

        x = x.view(pred_pairs_shape[0], -1)
       
        return x


           
class GNNPathContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    Path file
    [
        [word list],
        [path list],
        {pair: path index}    
    ]
    pairs =dict=> path_indexs =dict=> path_list =embedding_layer=>  path_embedding
    """
    def __init__(self, config, obj_classes, in_channels):
        super(GNNPathContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.hidden_dim = self.cfg.MODEL.EXRERNAL_KNOWLEDGE.PATH_DIM

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        class2idx = dict(zip(obj_classes, range(self.num_obj_classes)))


        neighbors = np.load(f'datasets/vg/VG_neighbor.npy', allow_pickle=True)

        neighbors = neighbors.item()
        entity_list = neighbors['entities']
        entity2idx = dict([(e,i) for i, e in enumerate(entity_list)])
        entity_neighbors = neighbors['neighbors']
        neighbors_embed_vecs = obj_edge_vectors(entity_list, wv_dir=self.cfg.GLOVE_DIR, wv_type='numberbatch-en', wv_dim=self.embed_dim)

        self.word2neighbor_embed = nn.Embedding(len(obj_classes), self.embed_dim)

        neigh_embedding = torch.zeros(len(obj_classes), self.embed_dim)


        for e, ns in entity_neighbors.items():
            n_emb = torch.Tensor(len(ns), self.embed_dim)
            for i, n in enumerate(ns):
                n_emb[i] = torch.mean(neighbors_embed_vecs[n], dim=0, keepdim=False)
            neigh_embedding[class2idx[entity_list[e]]] = torch.mean(n_emb, dim=0, keepdim=False)

        # with torch.no_grad():
        #     self.word2neighbor_embed.weight.copy_(neigh_embedding, non_blocking=True)

        self.entity2embed = nn.Embedding(len(obj_classes), self.embed_dim)

        entity_embed = torch.zeros(len(obj_classes), self.embed_dim)

        for i, obj in enumerate(obj_classes[1:]):
            entity_embed[i+1] = neighbors_embed_vecs[entity2idx[obj]]

        with torch.no_grad():
            self.entity2embed.weight.copy_(entity_embed, non_blocking=True)


        # self.neig2graph = nn.Linear(self.embed_dim*2+128, self.embed_dim+128)
 
        checkpoint = torch.load('datasets/vg/sub_graph_1.4.pth')
        nodes_dic = checkpoint['nodes_dic']
        idx2node = dict([(v, k) for k,v in nodes_dic.items()])
        pairs_dic = checkpoint['pairs_dic']
        graphs = checkpoint['graphs']
        max_n_label = checkpoint['max_n_label']
        node_information = checkpoint['node_information']

        node_information = torch.from_numpy(node_information)

        
        neigh_embedding = torch.zeros(len(idx2node), self.embed_dim)

        for obj in obj_classes:
            if obj in nodes_dic:
                neigh_embedding[nodes_dic[obj]] = neigh_embedding[class2idx[obj]]
            else:
                print(f'cannot find {obj}')


        center_node_list = []
        gnn_input = self.embed_dim+128
        for g in graphs:
            center_node_list.append(g.node_features[:2])
            g.node_features = node_information[g.node_features]
  
        self.geometric_batch = []


        for g in graphs:
            d = Data(x=g.node_features, edge_index=torch.from_numpy(g.edge_pairs.reshape((-1,2)).T).long())
            self.geometric_batch.append(d)

        self.using_neighbor = config.MODEL.EXRERNAL_KNOWLEDGE.GRAPH_WITH_NEIGHBOR
        self.pure_neighbor = config.MODEL.EXRERNAL_KNOWLEDGE.PURE_NEIGHBOR
        if self.using_neighbor:
        #     self.using_neighbor = True
        #     self.geometric_nei_batch = []
        #     for g, center_node in zip(graphs, center_node_list):
        #         d = Data(x=g.node_features.clone(), edge_index=torch.from_numpy(g.edge_pairs.reshape((-1,2)).T).long())
        #         d.x[0:2, -300:] = neigh_embedding[center_node]
        #         self.geometric_nei_batch.append(d)

        #     self.final = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        # else:

            for g, center_node in zip(self.geometric_batch, center_node_list):
                if self.pure_neighbor:
                    g.x[0:2, -300:] = neigh_embedding[center_node]
                else:
                    g.x[0:2, -300:] = (neigh_embedding[center_node]+ g.x[0:2, -300:])/2
                # self.geometric_nei_batch.append(d)
        self.final = nn.Linear(2*self.embed_dim+self.hidden_dim, self.hidden_dim)


        sortpooling_k = 0.6
        num_nodes_list = sorted([g.num_nodes for g in graphs])
        k_ = int(math.ceil(sortpooling_k * len(num_nodes_list))) - 1
        sortpooling_k = max(10, num_nodes_list[k_])
        print('k used in SortPooling is: ' + str(sortpooling_k))

        # self.gnn = Classifier(latent_dim=[32, 32, 32, 1], out_dim=self.hidden_dim, num_node_feats=128+300, num_edge_feats=0, sortpooling_k=sortpooling_k)
        if config.MODEL.EXRERNAL_KNOWLEDGE.GNN_MODE=="DGCNN":
            self.gnn = DGCNN(hidden_channels=64, num_layers=config.MODEL.EXRERNAL_KNOWLEDGE.GNN_LAYER, out_features=self.hidden_dim, k=sortpooling_k, num_features=gnn_input)
        elif config.MODEL.EXRERNAL_KNOWLEDGE.GNN_MODE=='GIN':
            self.gnn = GIN(hidden_channels=64, num_layers=config.MODEL.EXRERNAL_KNOWLEDGE.GNN_LAYER, out_features=self.hidden_dim, num_features=gnn_input)


        self.final_dic = {}

        for pair, v in pairs_dic.items():
            p = (class2idx[idx2node[pair[0]]], class2idx[idx2node[pair[1]]])
            self.final_dic[p] = v




    def forward(self, pred_pairs, union_features):
        l = []
        nei_list = []
        neighbor_embedding = self.word2neighbor_embed(pred_pairs.long()).cuda()
        for p, nei in zip(pred_pairs, neighbor_embedding):
            p = tuple(p.tolist())
            
            v = self.final_dic[p]

            l.append(self.geometric_batch[v])

            # if self.using_neighbor:
            #     nei_list.append(self.geometric_nei_batch[v])

        b = Batch.from_data_list(l)
        emb = self.gnn(x=b['x'].cuda(), batch=b['batch'].cuda(), edge_index=b['edge_index'].cuda())

        # if self.using_neighbor:
        #     b = Batch.from_data_list(nei_list)
        #     nei_emb = self.gnn(x=b['x'].cuda(), batch=b['batch'].cuda(), edge_index=b['edge_index'].cuda())

        #     x = self.final(torch.cat([emb, nei_emb], dim=1))
        # else:

        original_emb = self.entity2embed(pred_pairs.long()).cuda().view(-1, self.embed_dim*2)
        x = self.final(torch.cat([emb, original_emb], dim=1))
        return x

class GraphLabelLayer(nn.Module):
    def __init__(self, config, obj_classes, in_channels):
        super(GraphLabelLayer, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.lin = nn.Linear(4*(self.num_obj_cls-1), config.MODEL.EXRERNAL_KNOWLEDGE.PATH_DIM)
        score = torch.load('datasets/vg/pair2score.pth')
        class2idx = dict(zip(obj_classes, range(self.num_obj_cls)))

        self.score = {}

        for k,v in score.items():
            self.score[(class2idx[k[0]], class2idx[k[1]])] = np.max(np.array(v))
    
    def generate_embedding(self, entity, entitys_in_figure):
        emb = torch.zeros(self.num_obj_cls-1)
        for e in  entitys_in_figure:
            s1 = 0
            if (entity, e) in self.score:
                s1 = self.score[(entity, e)]
            s2 = 0
            if (e, entity) in self.score:
                s2 = self.score[(e, entity)]
            emb[e-1] = (s1+s2)/2
        return emb  

    def generate_position_embedding(self, entity):
        emb = torch.zeros(self.num_obj_cls-1)
        emb[entity-1] = 1
        return emb  


    def forward(self, pair_preds):

        contexts = []

        for pairs in pair_preds:
            pairs = pairs.cpu().numpy().astype(int)
            unique_pairs = np.unique(pairs)

            for pair in pairs:
            
                p0 = self.generate_embedding(pair[0], unique_pairs)

                p1 = self.generate_embedding(pair[1], unique_pairs)

                t0 = self.generate_position_embedding(pair[0])

                t1 = self.generate_position_embedding(pair[1])
                
                context = torch.cat((p0, p1, t0, t1))

                contexts.append(context.reshape((1, -1)))


        context = torch.cat(contexts).cuda()

        context = self.lin(context)

        return context 


class LabelLayer(nn.Module):
    def __init__(self, config, obj_classes, in_channels):
        super(LabelLayer, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.lin = nn.Linear(self.num_obj_cls-1, config.MODEL.EXRERNAL_KNOWLEDGE.PATH_DIM)
        word_list, pair_dict, position_dict = np.load(f'datasets/vg/VG_paths_{path_length}.npy', allow_pickle=True)


    def forward(self, pair_preds):

        context = []

        for pairs in pair_preds:

            tmp = torch.zeros((pairs.shape[0], self.num_obj_cls-1))
            pairs = np.unique(pairs.cpu().numpy().astype(int))
            for p in pairs:
                tmp[:, p-1] = 1
            context.append(tmp)

        context = torch.cat(context).cuda()

        context = self.lin(context)

        return context 
