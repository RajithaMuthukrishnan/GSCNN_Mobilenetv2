# GSCNN_Mobilenetv2
Tensorflow keras Gated-Shape CNN with MobileNetV2 backbone

Source code for Gated-Shape CNN with MobileNetV2 implementation.

Base Source code obtained from https://github.com/ben-davidson-6/Gated-SCNN


1. Dataset - gated_shape_cnn/cityscape
   Dataset images are not uploaded to this folder as Cityscape dataset[1] is not meant for distribution.
2. Dataset preparation files- gated_shape_cnn/datasets/cityscapes/ 
3. Training scripts (loss, train scripts, final dataset preparation before training) -  gated_shape_cnn/training/

Modified scripts :
1. MobileNetV2 definition - gated_shape_cnn/model/mobilenetv2.py
2. Shape stream layers, Fusion module layers definition - gated_shape_cnn/model/layers.py
3. GSCNN model definition - gated_shape_cnn/model/model_definition.py


Cityscapescripts[2] is installed for validation set evaluation


[1] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[2] Cityscape Evaluation scripts:  https://github.com/mcordts/cityscapesScripts


Further work on model definition is required to produce good results
