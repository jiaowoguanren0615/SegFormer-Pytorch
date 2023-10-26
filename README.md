# SegFormer-Pytorch
## Paper: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
https://arxiv.org/pdf/2105.15203.pdf

## Precautions
This code repository is a reproduction of the ___SegFormer-MiT___ model based on the Pytorch framework, including ___MiT-B0~B5___. It is currently only trained on the CityScape dataset. If you need to train your own dataset, please add a customized Mydataset in the ___datasets___ directory. class, and then modify the ___data_root___, ___batchsize___, ___epochs___ and other training parameters in the ___train_gpu.py___ file.

## TRAIN & EVALUATE MODEL
1. cd SegFormer
2. python train_gpu.py
