# Troubleshooting

Here is a compilation if common issues that you might face
while compiling / running this code:

## Compilation errors when compiling the library
If you encounter build errors like the following:
```
/usr/include/c++/6/type_traits:1558:8: note: provided for ‘template<class _From, class _To> struct std::is_convertible’
     struct is_convertible
        ^~~~~~~~~~~~~~
/usr/include/c++/6/tuple:502:1: error: body of constexpr function ‘static constexpr bool std::_TC<<anonymous>, _Elements>::_NonNestedTuple() [with _SrcTuple = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>&&; bool <anonymous> = true; _Elements = {at::Tensor, at::Tensor, at::Tensor, at::Tensor}]’ not a return-statement
     }
 ^
error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1
```
check your CUDA version and your `gcc` version.
```
nvcc --version
gcc --version
```
If you are using CUDA 9.0 and gcc 6.4.0, then refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/25,
which has a summary of the solution. Basically, CUDA 9.0 is not compatible with gcc 6.4.0.

## ImportError: No module named maskrcnn_benchmark.config when running webcam.py

This means that `maskrcnn-benchmark` has not been properly installed.
Refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/22 for a few possible issues.
Note that we now support Python 2 as well.


## ImportError: Undefined symbol: __cudaPopCallConfiguration error when import _C

This probably means that the inconsistent version of NVCC compile and your conda CUDAToolKit package. This is firstly mentioned in https://github.com/facebookresearch/maskrcnn-benchmark/issues/45 . All you need to do is:

```
# Check the NVCC compile version(e.g.)
/usr/cuda-9.2/bin/nvcc --version
# Check the CUDAToolKit version(e.g.)
~/anaconda3/bin/conda list | grep cuda

# If you need to update your CUDAToolKit
~/anaconda3/bin/conda install -c anaconda cudatoolkit==9.2
```

Both of them should have the **same** version. For example, if NVCC==9.2 and CUDAToolKit==9.2, this will be fine while when NVCC==9.2 but CUDAToolKit==9, it fails.


## Segmentation fault (core dumped) when running the library
This probably means that you have compiled the library using GCC < 4.9, which is ABI incompatible with PyTorch.
Indeed, during installation, you probably saw a message like
```
Your compiler (g++ 4.8) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
```
Follow the instructions on https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
to install GCC 4.9 or higher, and try recompiling `maskrcnn-benchmark` again, after cleaning the
`build` folder with
```
rm -rf build
```

## Result

### 1
2020-09-17 15:33:16,552 maskrcnn_benchmark INFO: 
====================================================================================================
Detection evaluation mAp=0.9995
====================================================================================================
SGG eval:     R @ 20: 0.5923;     R @ 50: 0.6556;     R @ 100: 0.6730;  for mode=predcls, type=Recall(Main).
SGG eval:  ng-R @ 20: 0.6738;  ng-R @ 50: 0.8183;  ng-R @ 100: 0.8889;  for mode=predcls, type=No Graph Constraint Recall(Main).
SGG eval:    zR @ 20: 0.0132;    zR @ 50: 0.0321;    zR @ 100: 0.0535;  for mode=predcls, type=Zero Shot Recall.
SGG eval: ng-zR @ 20: 0.0156; ng-zR @ 50: 0.0586; ng-zR @ 100: 0.1484;  for mode=predcls, type=No Graph Constraint Zero Shot Recall.
SGG eval:    mR @ 20: 0.1266;    mR @ 50: 0.1609;    mR @ 100: 0.1744;  for mode=predcls, type=Mean Recall.
----------------------- Details ------------------------
(above:0.1710) (across:0.0000) (against:0.0000) (along:0.0092) (and:0.0118) (at:0.2851) (attached to:0.0009) (behind:0.5896) (belonging to:0.0000) (between:0.0069) (carrying:0.2912) (covered in:0.1060) (covering:0.0226) (eating:0.2717) (flying in:0.0000) (for:0.0581) (from:0.0141) (growing on:0.0000) (hanging from:0.0478) (has:0.8091) (holding:0.7069) (in:0.3834) (in front of:0.1281) (laying on:0.0045) (looking at:0.0947) (lying on:0.0000) (made of:0.0000) (mounted on:0.0000) (near:0.4585) (of:0.6421) (on:0.8030) (on back of:0.0000) (over:0.0980) (painted on:0.0000) (parked on:0.0000) (part of:0.0000) (playing:0.0000) (riding:0.3426) (says:0.0000) (sitting on:0.3059) (standing on:0.0210) (to:0.0000) (under:0.3768) (using:0.1621) (walking in:0.0000) (walking on:0.1123) (watching:0.3031) (wearing:0.9699) (wears:0.0000) (with:0.1134) 
--------------------------------------------------------
SGG eval: ng-mR @ 20: 0.1932; ng-mR @ 50: 0.3243; ng-mR @ 100: 0.4421;  for mode=predcls, type=No Graph Constraint Mean Recall.
----------------------- Details ------------------------
(above:0.6453) (across:0.1032) (against:0.0726) (along:0.2125) (and:0.1617) (at:0.6790) (attached to:0.3916) (behind:0.6976) (belonging to:0.2567) (between:0.0938) (carrying:0.7367) (covered in:0.3369) (covering:0.3197) (eating:0.6323) (flying in:0.0000) (for:0.3348) (from:0.0282) (growing on:0.0920) (hanging from:0.4186) (has:0.9247) (holding:0.8554) (in:0.8123) (in front of:0.4551) (laying on:0.5233) (looking at:0.4277) (lying on:0.3141) (made of:0.0312) (mounted on:0.0677) (near:0.6874) (of:0.9151) (on:0.9414) (on back of:0.1844) (over:0.3592) (painted on:0.0891) (parked on:0.7670) (part of:0.0926) (playing:0.0000) (riding:0.9184) (says:0.0000) (sitting on:0.7490) (standing on:0.6222) (to:0.2377) (under:0.6367) (using:0.4548) (walking in:0.1338) (walking on:0.7991) (watching:0.3739) (wearing:0.9732) (wears:0.8542) (with:0.6920) 
--------------------------------------------------------
SGG eval:     A @ 20: 0.6898;     A @ 50: 0.6925;     A @ 100: 0.6925;  for mode=predcls, type=TopK Accuracy.
====================================================================================================
