# COACHER
This is all the code for our ECML-PKDD 2021 Paper: [Zero-Shot Scene Graph Relation Prediction through Commonsense Knowledge Integration](https://arxiv.org/abs/2107.05080). We developed this repository based on KaiHua Tang's [SGG Benchmark Project](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).


## Install

Please follow the instruction sin INSTALL.md to set up the environment.

## Data and Settings

We use the public dataset Visual Genome, with our zero-shot setting stored in [last_0.5_biased_zero_shot.pt](https://drive.google.com/file/d/17vhplu-RnupMMkCddDbVne0FbaTVqYTv/view?usp=sharing). After downloading this file, please place it to: `Coacher/maskrcnn_benchmark/data/datasets/evaluation/vg/`

The neighbor information is stored in file [VG_neighbor.npy](https://drive.google.com/file/d/1dBut1oF0GnKEcPcwvoCfcewQv1PrES-j/view?usp=sharing). After downloadining this file, please place it to: `Coacher/maskrcnn_benchmark/datasets/vg`

The path information is stored in file [sub_graph_1.4.pth](https://drive.google.com/file/d/1TWSPdsa-4A0i99QXbT--txrYyjQFTczN/view?usp=sharing). After download it please place it to this path `Coacher/maskrcnn_benchmark/datasets/vg`

Regarding [ConceptNet Embedding](https://github.com/commonsense/conceptnet-numberbatch). Please download the newest version and set the parameters `GLOVE_DIR` as the path where you store the downloaded embedding file.

## Usage

### Zero-shot dataset

#### Neighbor

```bash
python -m torch.distributed.launch --master_port 10303 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/N_last_0.5_without_biased_data_G+centeremb_1_layer MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.PATH False MODEL.EXRERNAL_KNOWLEDGE.ZERO_SHOT_PATH last_0.5_biased_zero_shot.pt

```

#### Path

```bash
python -m torch.distributed.launch --master_port 10302 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR GNNPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/G_last_0.5_without_biased_data_G+centeremb_1_layer MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.GRAPH_WITH_NEIGHBOR False MODEL.EXRERNAL_KNOWLEDGE.GNN_LAYER 1 MODEL.EXRERNAL_KNOWLEDGE.GNN_MODE DGCNN MODEL.EXRERNAL_KNOWLEDGE.ZERO_SHOT_PATH last_0.5_biased_zero_shot.pt

```

#### Neighbor+Path

```bash
python -m torch.distributed.launch --master_port 10301 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR GNNPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/N+G_last_0.5_without_biased_data_G+centeremb_1_layer MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.GRAPH_WITH_NEIGHBOR True MODEL.EXRERNAL_KNOWLEDGE.GNN_LAYER 1 MODEL.EXRERNAL_KNOWLEDGE.GNN_MODE DGCNN MODEL.EXRERNAL_KNOWLEDGE.ZERO_SHOT_PATH last_0.5_biased_zero_shot.pt

```

### The original dataset

#### Neighbor

```bash

python -m torch.distributed.launch --master_port 10303 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/N+G_original_without_biased_data MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.PATH False 

```

#### Path

```bash
python -m torch.distributed.launch --master_port 10304 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR GNNPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/G_original_without_biased_data_layer_1 MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.GRAPH_WITH_NEIGHBOR False 

```

#### Neighbor+Path

```bash
python -m torch.distributed.launch --master_port 10301 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR GNNPredictorWithExternalKnowledge SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 12 DTYPE "float32" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /local/home/{Your Name}/glove MODEL.PRETRAINED_DETECTOR_CKPT /local/home/{Your Name}/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /local/home/{Your Name}/checkpoints/N_original_without_biased_data_G+centeremb_1_layer MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.EXRERNAL_KNOWLEDGE.GRAPH_WITH_NEIGHBOR True 

```

## Citation

Please cite our paper if you find this code useful:

```
@misc{kan2021zeroshot,
      title={Zero-Shot Scene Graph Relation Prediction through Commonsense Knowledge Integration}, 
      author={Xuan Kan and Hejie Cui and Carl Yang},
      year={2021},
      eprint={2107.05080},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```