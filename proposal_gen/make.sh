# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
# Compute/shader model   Cards
# 7.0                    V100, Titan V
# 6.1                    P4, P40, Titan Xp, GTX 1080 Ti, GTX 1080
# 6.0                    P100
# 5.2                    M40, Titan X, GTX 980
# 3.7                    K80
# 3.5                    K40, K20
# 3.0                    K10, Grid K520 (AWS G2)
##### Please change this accordingly ########
arch = 'sm_61' 
cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=$arch
cd ../../
python build.py
cd ../

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=$arch
cd ../../
python build.py
cd ../../