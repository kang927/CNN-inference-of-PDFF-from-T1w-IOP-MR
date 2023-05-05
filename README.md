# CNN-inference-of-PDFF-from-T1w-IOP-MR
# Python code for training a CNN to infer PDFF maps from T1w IOP MRI 
# The codes are used in a jupyter notebook and try to be self contained by puting alot of the helper function loaded before the main training routine
# Only requirement are: Tensorflow 2.0/Keras 2.0,  Tensorlayer 
# The first part of the program (before line 615) is for training the CNN
# Then, after line 616, it is cross-validation for evaluating the model and computing various statistic such as ICC, regression, and Bland-Altman
# 
