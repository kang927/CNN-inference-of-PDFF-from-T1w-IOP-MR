#!/usr/bin/env python
# coding: utf-8

# In[1]:


# limiting the usage of memory 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# In[2]:


import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model
# for cross-validation
from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import pandas as pd

import pingouin as pg # for compute ICC 

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[3]:


from os import listdir
import os
#from matplotlib.pyplot import imshow
import sys, re
import matplotlib.pyplot as plt
import nibabel as nib
# path for private modules
sys.path.append('/tf/deeplearning/mri_liver_seg')

# hierarchal data HDF5 format
import h5py

from src_tf2.utility import imshow, transform_ITKSNAP_to_training
#from src.train import get_subset, hist_match,batch_read, batch_read_wt_histeq
from src_tf2.process_dicom import rescale_image
# load the model of interest
from src_tf2.models import get_unet_with_STN, get_unet, get_unet_with_deformconv
from src_tf2.metrics import dice_coef_loss, dice_coef, jacc_dist, convert_binary, postprocess_cnn


from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter


# In[4]:


# get the input and output according to case_lst
# training_flag - if data is for training, then not output the segmentation mask 
# img_normalization, whether to normalize each case or all the case together  
def get_X_Y(hd5_fn,case_lst, liver_area_th=100, r2star_max = 250, training_flag=True, 
            img_normalization='batch', img_scaling_factor=1700):
    
    num_cases = len(case_lst)

    results = dict()
    # must be multiple of 2 
    nx = 176
    ny = 208
    Nx = 224
    Ny = 224
    dx = np.int( (Nx - nx)/2 )
    dy = np.int( (Ny - ny)/2 )
    n_channels = 2
    # data is in format: n_channels, nx, ny, nz 
    sigma = [0.2,0.2,0.001]

    for case_index in range(0,num_cases):
        case_num = case_lst[case_index]
        print(f"case: {case_num}")
        with h5py.File(hd5_fn,'r') as f:

            lava_ip = np.double( f[case_num + '/lava_ip_reg'] )
            lava_op = np.double( f[case_num + '/lava_op_reg'] )             
            tmp_img = np.double( f[case_num + '/idealiq_pdffmap' ] )   
            tmp_img2 = np.double( f[case_num + '/idealiq_r2starmap' ] )   
 
            if tmp_img.shape[0] != Nx:
                print("Warning for case: " + str(case_num))
                print(f"pdff map size not {Nx}, rescale to fit")
                tmp_img = resize(tmp_img,(Nx,Ny,tmp_img.shape[2]),anti_aliasing=True)
                tmp_img2 = resize(tmp_img2,(Nx,Ny,tmp_img2.shape[2]),anti_aliasing=True)
                
            if lava_ip.shape[0] != Nx:
                print("Warning for case: " + str(case_num))
                print(f"lava images size not {Nx}, rescale to fit")
                lava_ip = resize(lava_ip,(Nx,Ny,lava_ip.shape[2]),anti_aliasing=True)
                lava_op = resize(lava_op,(Nx,Ny,lava_op.shape[2]),anti_aliasing=True)
                
            # need to make sure lava_ip/op and pdff have same number of slices
            if (lava_ip.shape[2] == tmp_img.shape[2]):
                # load segmentation images
                lava_segmask = np.double( f[case_num + '/lava_reg_segmask'] )
                idealiq_segmask = np.double( f[case_num + '/idealiq_segmask'] )

                Nz = tmp_img.shape[2]         
                lava_liver_area = np.sum(lava_segmask,axis = (0,1))
                idealiq_liver_area = np.sum(idealiq_segmask,axis = (0,1))
                if len(lava_liver_area) == len(idealiq_liver_area):
                    z_index = (idealiq_liver_area >= liver_area_th) & (lava_liver_area >= liver_area_th)
                else:
                    z_index = (lava_liver_area >= liver_area_th)


                Nz = tmp_img.shape[2]            
                lava_ip_blur = gaussian_filter(lava_ip[dx:Nx-dx,dy:Ny-dy,z_index],sigma)
                lava_op_blur = gaussian_filter(lava_op[dx:Nx-dx,dy:Ny-dy,z_index],sigma)

                pdff = tmp_img[dx:Nx-dx,dy:Ny-dy,z_index]
                r2star = tmp_img2[dx:Nx-dx,dy:Ny-dy,z_index]
                # get rid of the pdff values when the input is just noise
                pdff[lava_ip_blur < 100] = 0.0
                pdff = pdff/100.0
                pdff = np.clip(pdff,0,1)
                
                # normalize r2 star value according to field strength
                if re.search(r'Arizona',case_num):
                    # since Arizona data are from 3.0 T, need to divide by 2 to correspond to 1.5T data in Okahoma
                    print("case from Arizona, normalize to 3T")
                    r2star = r2star/r2star_max/2 
                else:
                    r2star = r2star/r2star_max   
                #r2start = np.clip(r2star,0,1)

                # number of slice for case: case_index
                nz_case = np.sum(z_index)
                if nz_case !=0:
                    if case_index == 0:
                       # x_train has the dimension n_channels x Nx x Ny x Nslices
                        tmp_img = np.append(np.array([lava_ip_blur]),np.array([lava_op_blur]),axis=0)
                        if img_normalization =='instance':
                            print("normalize each case individiually!")
                            tmp_img = rescale_image(tmp_img, low_p=5,up_p=95)

                        x_train = tmp_img
                        y_train = np.append(np.array([pdff]),np.array([r2star]),axis=0)
                        if not training_flag:                        
                            case_indices = np.ones([nz_case,])*case_index
                        lava_segmask_lst = lava_segmask[dx:Nx-dx,dy:Ny-dy,z_index]
                        idealiq_segmask_lst = idealiq_segmask[dx:Nx-dx,dy:Ny-dy,z_index]
                    else:
                        tmp_img = np.append(np.array([lava_ip_blur]),np.array([lava_op_blur]),axis=0)
                        if img_normalization =='instance':
                            tmp_img = rescale_image(tmp_img, low_p=5,up_p=95)

                        tmp2 = np.append(np.array([pdff]),np.array([r2star]),axis=0)
                        # stack along the z-axis 
                        x_train = np.append(x_train,tmp_img,axis=3)
                        y_train = np.append(y_train,tmp2,axis=3)
                        if not training_flag:
                            tmp_case_index = np.ones([nz_case,])*case_index
                            case_indices = np.append( case_indices, tmp_case_index, axis=0 )
                        lava_segmask_lst = np.append(lava_segmask_lst,lava_segmask[dx:Nx-dx,dy:Ny-dy,z_index],axis=2)
                        idealiq_segmask_lst = np.append(idealiq_segmask_lst,idealiq_segmask[dx:Nx-dx,dy:Ny-dy,z_index],axis=2)
                else:
                    print(f"Case: {case_num} does not meet liver_area threshold!")

                    
            else: # mismatch, so ignore this case
                print("Warning for case: " + str(case_num))
                print(f"lava and idealiq images don't have same slices, not using as training data")
    
    # reorder the structures so the # of channels is the last element
    x_train = np.array(np.transpose(x_train,[3,1,2,0]),dtype=np.float32)
    y_train = np.array(np.transpose(y_train,[3,1,2,0]),dtype=np.float32)
    # normalize the input data as a whole 
    if img_normalization == 'batch':
        print("normalize all the case as a batch together")
        #X_train_norm = np.clip( rescale_image(tmp_img, low_p=5,up_p=95),0, 5)
        X_train_norm = np.clip(x_train/img_scaling_factor,0,10)
        
    else:
        X_train_norm = np.clip(x_train,0,5)

    Y_train_norm = y_train
    
    # some images are all zero, probably due to process of registration, need to delete them 
    # since they are not relevant to evaluate the model 
    tmp1 = np.sum(X_train_norm,axis=(1,2,3))
    index = np.where(tmp1==0)
    tmp = np.delete(X_train_norm,index[0],0)
    X_train_clean = tmp
    tmp = np.delete(Y_train_norm,index[0],0)
    Y_train_clean = tmp
    
    # if training, then we only need to return input_data and output_data
    results['input_data'] = X_train_clean
    results['output_data'] = Y_train_clean
    
    # if testing, we need to return additionally the indices corresponding to each case and the segmentation mask
    # for computation of average pdff/r2star etc. for each case
    if not training_flag:
        results['case_indices'] = case_indices
        results['lava_segmask'] = lava_segmask_lst
        results['idealiq_segmask'] = idealiq_segmask_lst
    
    tmp_loss_weight = np.ones(lava_segmask_lst.shape)*0.1
    tmp_loss_weight[ idealiq_segmask_lst==1 ] = 5
    results['loss_weight'] = np.array(tmp_loss_weight,dtype = np.float32)
    
    return results


# In[5]:


# define custom loss function 
def custom_loss(y_pred, y_true):
    w1 = 0.05
    w2 = 0.95
    assert(y_pred.shape == y_true.shape)
    y_shape = y_pred.shape
    y_pred_flat = tf.reshape(y_pred, [-1, y_shape[1] * y_shape[2] * y_shape[3]])
    y_true_flat = tf.reshape(y_true, [-1, y_shape[1] * y_shape[2] * y_shape[3]]) 

    # scale the cosine loss to be range (0,2) instead of (-1,1) and maximize cosine
    loss1 = tf.reduce_mean(-1*tl.cost.cosine_similarity(y_pred_flat, y_true_flat) + 1)
    #loss2 = tl.cost.absolute_difference_error(y_pred, y_true)
    loss2 = tl.cost.mean_squared_error(y_pred, y_true)
    loss_custom = w1*loss1 + w2*loss2
    
    return loss_custom


# In[6]:


def train_model(model,X_train,Y_train,X_test,Y_test,n_epoch,batch_size,print_freq,optimizer):
    net = model
    train_weights = net.trainable_weights

    print("Training ...")
    for epoch in range(n_epoch):
        start_time = time.time()

        net.train()  # enable dropout

        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, Y_train, batch_size, shuffle=True):
            # input_dim must be of length 4
            #X_train_list = [X_train_a[:,:,:,0:1],X_train_a[:,:,:,1:2]]

            with tf.GradientTape() as tape:
                ## compute outputs
                _logits, _ = net(X_train_a)  # alternatively, you can use MLP(x, is_train=True) and remove MLP.train()
                ## compute loss and update model
                _loss = custom_loss(_logits, y_train_a) #+ 
                
            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

        ## use training and evaluation sets to evaluate the model every print_freq epoch
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            net.eval()  # disable dropout
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_iter = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, Y_train, batch_size, shuffle=False):
                #X_train_list = [X_train_a[:,:,:,0:1],X_train_a[:,:,:,1:2]]
                _logits, _ = net(X_train_a)  # alternatively, you can use MLP(x, is_train=False) and remove MLP.eval()
                train_loss = train_loss + custom_loss(_logits, y_train_a)
                n_iter += 1
            print("   train loss: %f" % (train_loss / n_iter))
            
            
            test_loss, test_acc, n_iter = 0, 0, 0
            for X_test_a, y_test_a in tl.iterate.minibatches(X_test, Y_test, batch_size, shuffle=False):
                #X_train_list = [X_train_a[:,:,:,0:1],X_train_a[:,:,:,1:2]]
                _logits, _ = net(X_test_a)  # alternatively, you can use MLP(x, is_train=False) and remove MLP.eval()
                test_loss = test_loss + custom_loss(_logits, y_test_a)
                n_iter += 1
            print("   test loss: %f" % (test_loss / n_iter))
            
    return net


# In[7]:


def eval_model_batch(model,X,batch_size):
    net = model
    net.eval()
    for ii in range(0,X.shape[0],batch_size):
        if ii == 0:
            pred, _ = net(X[ii:ii+batch_size,:,:,:])
            Pred = pred
        else:
            pred, _ = net(X[ii:ii+batch_size,:,:,:])
            Pred = np.append(Pred,pred,axis=0)
    return Pred


# In[8]:


# second stage of training
# use the model that predict rFF as an initial model
# directly learn to predict actual PDFF from lava IP and OP based on the manually aligned dataset
#hd5_fn = '/data/tmp/lavaIPOP_to_idealiqpdff_SHSdataset_combined.hdf5'

# hd5_fn with liver registration
hd5_fn = '/data/tmp/lavaIPOP_to_idealiqpdff_SHSdataset_liver_reg.hdf5'
group1 = '/data2/Arizona'
group2 = '/data2/Oklahoma'

# create the case_lst directory
case_lst_AZ = list()
case_lst_OKA = list()


with h5py.File(hd5_fn,'r') as f:
    num_cases_AZ = len(f[group1].keys())
    num_cases_OKA = len(f[group2].keys())
    for key in f[group1].keys():
        case_lst_AZ.append(group1 + '/' + key)
    
    for key in f[group2].keys():
        case_lst_OKA.append(group2 + '/' + key)


print(f"num of cases for AZ: {num_cases_AZ}")
print(f"num of cases for OKA: {num_cases_OKA}")


# In[9]:



with h5py.File(hd5_fn,'r') as f:
    for key in f['/data2/Arizona/017 SHS36093608152018I'].keys():
        print(key)


# In[10]:


# preparing training and testing data
# split each institution data into 75%/25% split and ensure r2* <100 and > 100 are evenly splitted 
r2star_max = 400
liver_area_th = 500

# AZ dataset
test_cases = get_X_Y(hd5_fn,case_lst_AZ, 
                       liver_area_th = liver_area_th, 
                       r2star_max = r2star_max,
                        img_normalization='batch',
                        training_flag = False)

Y_test = test_cases['output_data']
case_index_lst = np.unique(test_cases['case_indices'])
num_test_cases = len(case_index_lst)
# calculate median R2* values for each case
pdff_liver_median_6echo_AZ = np.zeros([num_test_cases,])
r2star_liver_median_6echo_AZ = np.zeros([num_test_cases,])  
ct = 0
for case_index in case_index_lst:
    test_case_Y = test_cases['output_data'][ test_cases['case_indices'] == case_index,:,:,: ]
    idealiq_segmask = test_cases['idealiq_segmask'][ :,:, test_cases['case_indices'] == case_index]

    # computing various parametric maps
    pdffmap_true = np.transpose(test_case_Y[:,:,:,0],[1,2,0])
    r2starmap_true = np.transpose(test_case_Y[:,:,:,1],[1,2,0])
    # compute median
    tmp = pdffmap_true[idealiq_segmask==1]
    pdff_liver_median_6echo_AZ[ct] = np.median(tmp)
    tmp = r2starmap_true[idealiq_segmask==1]
    r2star_liver_median_6echo_AZ[ct] = np.median(tmp)
    ct+=1

pdff_liver_median_6echo_AZ = pdff_liver_median_6echo_AZ*100
r2star_liver_median_6echo_AZ = r2star_liver_median_6echo_AZ*r2star_max*2

# categorize the r2star so they split between training and testing
r2star_group_AZ = np.zeros([num_test_cases,], dtype=np.int32)
r2star_group_AZ[r2star_liver_median_6echo_AZ >= 100] = 1


# OKA dataset
test_cases = get_X_Y(hd5_fn,case_lst_OKA, 
                       liver_area_th = liver_area_th, 
                       r2star_max = r2star_max,
                        img_normalization='batch',
                        training_flag = False)

Y_test = test_cases['output_data']
case_index_lst = np.unique(test_cases['case_indices'])
num_test_cases = len(case_index_lst)
# calculate median R2* values for each case
pdff_liver_median_6echo_OKA = np.zeros([num_test_cases,])
r2star_liver_median_6echo_OKA = np.zeros([num_test_cases,])  
ct = 0
for case_index in case_index_lst:
    test_case_Y = test_cases['output_data'][ test_cases['case_indices'] == case_index,:,:,: ]
    idealiq_segmask = test_cases['idealiq_segmask'][ :,:, test_cases['case_indices'] == case_index]

    # computing various parametric maps
    pdffmap_true = np.transpose(test_case_Y[:,:,:,0],[1,2,0])
    r2starmap_true = np.transpose(test_case_Y[:,:,:,1],[1,2,0])
    # compute median
    tmp = pdffmap_true[idealiq_segmask==1]
    pdff_liver_median_6echo_AZ[ct] = np.median(tmp)
    tmp = r2starmap_true[idealiq_segmask==1]
    r2star_liver_median_6echo_OKA[ct] = np.median(tmp)
    ct+=1

pdff_liver_median_6echo_OKA = pdff_liver_median_6echo_OKA*100
# OKA = 1.5T so don't need to multiple by scaling factor of 2
r2star_liver_median_6echo_OKA = r2star_liver_median_6echo_OKA*r2star_max
r2star_group_OKA = np.zeros([num_test_cases,], dtype=np.int32)
r2star_group_OKA[r2star_liver_median_6echo_OKA >= 50] = 1


# In[11]:


case_lst_AZ.remove('/data2/Arizona/SHS36110801302019I')
case_lst_AZ.remove('/data2/Arizona/SHS36004808292018I')


# In[12]:


# splitting the AZ dataset
n_splits = 4
X = case_lst_AZ
y = r2star_group_AZ
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
train_index_AZ, test_index_AZ = next(skf.split(X, y))
print("For the AZ dataset: training and test split: ")
print('train -  {}   |   test -  {}'.format(np.bincount(y[train_index_AZ]), np.bincount(y[test_index_AZ])))


# splitting OKA dataset
X = case_lst_OKA
y = r2star_group_OKA

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
train_index_OKA, test_index_OKA = next(skf.split(X, y))
print("For the OKA dataset: training and test split: ")
print('train -  {}   |   test -  {}'.format(np.bincount(y[train_index_OKA]), np.bincount(y[test_index_OKA])))


# In[13]:


# want to even split those with high R2* between training and testing
r2star_AZ_index = np.where(r2star_liver_median_6echo_AZ>=100)
print(r2star_AZ_index[0])
print(r2star_liver_median_6echo_AZ[r2star_AZ_index[0]])

r2star_OKA_index = np.where(r2star_liver_median_6echo_OKA >= 100)
print(r2star_OKA_index[0])
print(r2star_liver_median_6echo_OKA[r2star_OKA_index[0]])


# In[14]:


# empirical encode the high R2* cases into testing set
# and remove it from the training set 
tmp = np.setdiff1d(train_index_AZ,[67, 102])
train_index_AZ = tmp
tmp = np.union1d(test_index_AZ,[67, 102])
test_index_AZ = tmp
print(r2star_liver_median_6echo_AZ[test_index_AZ])


tmp = np.setdiff1d(train_index_OKA,[9])
train_index_OKA = tmp
tmp = np.union1d(test_index_OKA,[9])
test_index_OKA = tmp
print(r2star_liver_median_6echo_OKA[test_index_OKA])


# In[15]:


print(np.intersect1d(train_index_AZ, test_index_AZ))
print(np.intersect1d(train_index_OKA, test_index_OKA))


# In[ ]:


plt.hist(r2star_liver_median_6echo_AZ[train_index_AZ])
plt.hist(r2star_liver_median_6echo_AZ[test_index_AZ])


# In[ ]:


plt.hist(r2star_liver_median_6echo_OKA[train_index_OKA])
plt.hist(r2star_liver_median_6echo_OKA[test_index_OKA])


# In[16]:


case_lst_combined = np.append(train_index_AZ,train_index_OKA, axis=0)
print(case_lst_combined)


# In[17]:


num_case_AZ = len(case_lst_AZ)
train_index_combined = np.append(train_index_AZ,train_index_OKA + num_case_AZ ,axis=0)
test_index_combined = np.append(test_index_AZ,test_index_OKA + num_case_AZ ,axis=0)
training_index = [train_index_AZ, num_case_AZ + train_index_OKA, train_index_combined]
test_index = [test_index_AZ, num_case_AZ + test_index_OKA, test_index_combined]
num_test_cases = len(test_index_combined)
print(training_index)
print(test_index)

case_lst_combined = case_lst_AZ + case_lst_OKA
print(case_lst_combined)
#print(test_site_index)


# In[18]:


test_site_index = np.ones([num_test_cases,])
test_site_index[len(test_index_AZ):] = 2
print(test_site_index)
print(len(test_site_index))
print(len(test_index_combined))


# In[ ]:


n_kfold = 3
r2star_max = 400
liver_area_th = 1000
for kfold in range(0,n_kfold):
    print(f"Performing {kfold} out of {n_kfold} cross-validation")
    training_lst = [case_lst_combined[ii] for ii in training_index[kfold]]
    test_lst = [case_lst_combined[ii] for ii in test_index[kfold]]
    
    print("loading training cases")
    training_cases = get_X_Y(hd5_fn,training_lst, 
                           liver_area_th = liver_area_th, 
                           r2star_max = r2star_max,
                            img_normalization='batch',
                            training_flag = True)
    
    X_train = training_cases['input_data']
    Y_train = training_cases['output_data']
    for ii in range(0,100,50):
        imshow(X_train[ii,:,:,0],X_train[ii,:,:,1],
               Y_train[ii,:,:,0], Y_train[ii,:,:,1])
    
    
    print("loading test cases")
    test_cases = get_X_Y(hd5_fn,test_lst, 
                       liver_area_th=liver_area_th,
                       r2star_max = r2star_max,
                       img_normalization='batch',
                       training_flag=False)
    
    
    X_test = test_cases['input_data']
    Y_test = test_cases['output_data']
    for ii in range(0,100,50):
        imshow(X_test[ii,:,:,0], X_test[ii,:,:,1],
               Y_test[ii,:,:,0],Y_test[ii,:,:,1],)
    
    # create the CNN model 
    nx = 176
    ny = 208
    Nx = 256
    Ny = 256
    dx = np.int( (Nx - nx)/2 )
    dy = np.int( (Ny - ny)/2 )

    # get the model of interest
    net = get_unet([None,nx, ny, 2],n_out=2)
    # setup training parameters
    n_epoch = 100
    learning_rate = 1e-4
    print_freq = 10
    batch_size = 8
    train_weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    net = train_model(net,X_train,Y_train,X_test,Y_test, n_epoch,batch_size,print_freq,optimizer)
    
    
    # save the model 
    print(f"Saving model file")
    model_weight_file = '/data/tmp/fixed_split_each_institute/model_lavaIPOP_to_idealiq_pdffandr2star_0.95L2_0.05cosine_model' + str(kfold+1) +  ".hdf5"
    net.save_weights(model_weight_file)


# ## Evaluate the model on test dataset as follow:
# ##### 1) Compute CNN and dixon predict PDFF maps and r2star maps 
# ##### 2) Compute mean and median liver pdff for each case  

# In[19]:


# evaluate on the hold-out test set for each cross-validation model
n_kfold = 3
r2star_max = 400
liver_area_th = 1000
delta = 1e-5
batch_size = 8


case_data_lst = list()

for kfold in range(0,n_kfold):    
    print(f"Evaluating performance on {kfold} out of {n_kfold} cross-validation test dataset")
    test_lst = [case_lst_combined[ii] for ii in test_index[2]]
    
    print("loading test cases")
    test_cases = get_X_Y(hd5_fn,test_lst, 
                       liver_area_th=liver_area_th,
                       r2star_max = r2star_max,
                       img_normalization='batch',
                       training_flag=False)
    
    # loading the saved cnn model
    print(f"Loading the model file")
    model_weight_file = '/data/tmp/fixed_split_each_institute/model_lavaIPOP_to_idealiq_pdffandr2star_0.95L2_0.05cosine_model' + str(kfold+1) +  ".hdf5"

    
    # create the CNN model 
    nx = 176
    ny = 208
    net = get_unet([None,nx, ny, 2],n_out=2)
    # load the model weights
    net.load_weights(model_weight_file)
    
    # evaluate the test set 
    input_images = test_cases['input_data']
    print(input_images.shape)
    Pred = eval_model_batch(net,input_images,batch_size)
    
    # get the number of cases for the current hold out test set  
    case_index_lst = np.unique(test_cases['case_indices'])
    num_test_cases = len(case_index_lst)
    
    # index which trained model used
    kfold_index = np.ones([num_test_cases,])*kfold
    pdff_liver_mean_true = np.zeros([num_test_cases,])
    pdff_liver_mean_cnn = np.zeros([num_test_cases,])
    pdff_liver_mean_dixon = np.zeros([num_test_cases,])
    pdff_liver_median_true = np.zeros([num_test_cases,])
    pdff_liver_median_cnn = np.zeros([num_test_cases,])
    pdff_liver_median_dixon = np.zeros([num_test_cases,])
    r2star_liver_mean_true = np.zeros([num_test_cases,])
    r2star_liver_mean_cnn = np.zeros([num_test_cases,])
    r2star_liver_median_true = np.zeros([num_test_cases,])
    r2star_liver_median_cnn = np.zeros([num_test_cases,])
    
    # evaluate on the test set case by case
    # evaluate on the test set case by case
    ct = 0
    for case_index in case_index_lst:
        case_num = test_lst[int(case_index)]
        print(f"case_num: {case_num}")
        delta = 1e-5
        case_data = dict()
        #print(f"processing case: {case_index}")
        test_case_X = test_cases['input_data'][ test_cases['case_indices'] == case_index,:,:,: ]
        test_case_Y = test_cases['output_data'][ test_cases['case_indices'] == case_index,:,:,: ]
        test_case_Pred = Pred[ test_cases['case_indices'] == case_index,:,:,: ] 
        lava_segmask = test_cases['lava_segmask'][ :,:, test_cases['case_indices'] == case_index]
        idealiq_segmask = test_cases['idealiq_segmask'][ :,:, test_cases['case_indices'] == case_index ]
        lava_segmask = idealiq_segmask
        
        case_data['input_data'] = test_case_X
        case_data['output_data'] = test_case_Y
        case_data['pred_data'] = test_case_Pred
        case_data['lava_segmask'] = lava_segmask
        case_data['idealiq_segmask'] = idealiq_segmask
        case_data_lst.append(case_data)

        # computing various parametric maps
        pdffmap_cnn_pred = np.transpose(test_case_Pred[:,:,:,0],[1,2,0])
        r2starmap_cnn_pred = np.transpose(test_case_Pred[:,:,:,1],[1,2,0])
        pdffmap_test = np.transpose(test_case_Y[:,:,:,0],[1,2,0])
        r2starmap_test = np.transpose(test_case_Y[:,:,:,1],[1,2,0])
        ip_test = np.transpose(test_case_X[:,:,:,0],[1,2,0])
        op_test = np.transpose(test_case_X[:,:,:,1],[1,2,0])

        Pdff_liver_true = pdffmap_test*idealiq_segmask
        R2starmap_liver_true = r2starmap_test*idealiq_segmask
        Pdff_liver_cnn = pdffmap_cnn_pred*lava_segmask
        R2starmap_liver_cnn = r2starmap_cnn_pred*lava_segmask
        Pdff_liver_dixon = (ip_test - op_test)/2.0/(ip_test + delta)*lava_segmask
        pdff_liver_mean_true[ct] = np.sum(Pdff_liver_true) / ( np.sum(idealiq_segmask) + delta )
        r2star_liver_mean_true[ct] = np.sum(R2starmap_liver_true) / ( np.sum(idealiq_segmask) + delta )
        pdff_liver_mean_cnn[ct] = np.sum(Pdff_liver_cnn) / ( np.sum(lava_segmask) + delta )
        r2star_liver_mean_cnn[ct] = np.sum(R2starmap_liver_cnn) / ( np.sum(lava_segmask) + delta )
        pdff_liver_mean_dixon[ct] = np.sum(Pdff_liver_dixon) / ( np.sum(lava_segmask) + delta )
        
        # compute median
        tmp = pdffmap_test[idealiq_segmask==1]
        pdff_liver_median_true[ct] = np.median(tmp)
        tmp = pdffmap_cnn_pred[lava_segmask==1]
        pdff_liver_median_cnn[ct] = np.median(tmp)
        pdffmap_dixon = (ip_test - op_test)/2.0/(ip_test + delta)
        tmp = pdffmap_dixon[lava_segmask==1]
        pdff_liver_median_dixon[ct] = np.median(tmp)
        tmp = r2starmap_test[idealiq_segmask==1]
        # we need to multiple a factor of 2 for 3.0 T studies to have the appropriate normalization factor
        if re.search(r'Arizona',case_num):
            r2star_liver_median_true[ct] = np.median(tmp)*2
            tmp = r2starmap_cnn_pred[lava_segmask==1]
            r2star_liver_median_cnn[ct] = np.median(tmp)*2
        else:
            r2star_liver_median_true[ct] = np.median(tmp)
            tmp = r2starmap_cnn_pred[lava_segmask==1]
            r2star_liver_median_cnn[ct] = np.median(tmp)
        ct += 1
    
    # use a pandas dataframe to hold all the result from cross-validation
    cross_val_data = {'kfold_index': kfold_index,
       'pdff_liver_mean_true': pdff_liver_mean_true,
       'pdff_liver_mean_cnn': pdff_liver_mean_cnn,
       'pdff_liver_mean_dixon': pdff_liver_mean_dixon,
       'r2star_liver_mean_true': r2star_liver_mean_true,
       'r2star_liver_mean_cnn': r2star_liver_mean_cnn,
       'pdff_liver_median_true': pdff_liver_median_true,
       'pdff_liver_median_cnn': pdff_liver_median_cnn,
       'pdff_liver_median_dixon': pdff_liver_median_dixon,
       'r2star_liver_median_true': r2star_liver_median_true,
       'r2star_liver_median_cnn': r2star_liver_median_cnn
                     }
    if kfold ==0:
        df_cross_val = pd.DataFrame(data=cross_val_data)
    else:
        df_tmp = pd.DataFrame(data=cross_val_data)
        df_cross_val = df_cross_val.append(df_tmp, ignore_index=True)
        


# In[20]:


def compute_ICC(data1, data2):
    nz = len(data1)
    method_type = ['method1']*nz + ['method2']*nz
    method_val = np.append( data1,data2)
    subject_id = np.append(np.arange(0,nz), np.arange(0,nz))
    eval_method = {'subject_id': subject_id,
               'method_type': method_type,
               'method_val':method_val}
    df_method_cmp = pd.DataFrame(data=eval_method)

    icc_result = pg.intraclass_corr(data=df_method_cmp, targets='subject_id',
                                        raters='method_type',
                                        ratings='method_val')
    return icc_result
    


# In[21]:


# computing the result and put it as a table 

subset_perf = list()
for subset_index in range(0,3):
    index = df_cross_val.kfold_index == subset_index
    x = df_cross_val.pdff_liver_median_cnn[index]*100
    y = df_cross_val.pdff_liver_median_true[index]*100
    # ICC
    icc_result = compute_ICC(x,y)
    # linear regresssion
    A = np.vstack([x,np.ones(x.shape[0])]).T
    m,c = np.linalg.lstsq(A,y)[0]
    # bland altman
    diff_val = (y - x)
    md = np.mean(diff_val)       # Mean of the difference
    sd = np.std(diff_val, axis=0) 
    LoA_low = md - 1.96*sd
    LoA_high = md + 1.96*sd
    perf_metrics ={'subset_index': subset_index,
                   'icc': icc_result['ICC'][1],
                   'slope': m,
                   'intercept': c,
                   'bias': md,
                   'LoA_low': LoA_low,
                   'LoA_high': LoA_high}
    subset_perf.append(perf_metrics)


# In[ ]:


[print(perf) for perf in subset_perf]


# In[ ]:


# computing the result and put it as a table 

subset_perf = list()
for subset_index in range(0,4):
    index = df_cross_val.kfold_index == subset_index
    x = df_cross_val.r2star_liver_median_cnn[index]*r2star_max
    y = df_cross_val.r2star_liver_median_true[index]*r2star_max
    # ICC
    icc_result = compute_ICC(x,y)
    # linear regresssion
    A = np.vstack([x,np.ones(x.shape[0])]).T
    m,c = np.linalg.lstsq(A,y)[0]
    # bland altman
    diff_val = (y - x)
    md = np.mean(diff_val)       # Mean of the difference
    sd = np.std(diff_val, axis=0) 
    LoA_low = md - 1.96*sd
    LoA_high = md + 1.96*sd
    perf_metrics ={'subset_index': subset_index,
                   'icc': icc_result['ICC'][1],
                   'slope': m,
                   'intercept': c,
                   'bias': md,
                   'LoA_low': LoA_low,
                   'LoA_high': LoA_high}
    subset_perf.append(perf_metrics)


# In[ ]:


[print(perf) for perf in subset_perf]


# In[22]:


# plotting true pdff vs. predicted pdff

subset_index = 2
index = df_cross_val.kfold_index == subset_index
# median hepatic values rather than mean
fontsize = 16
N_sub = 147

x = df_cross_val.pdff_liver_median_cnn[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


icc_result = compute_ICC(x_siteA,y_siteA)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x_siteA,np.ones(x_siteA.shape[0])]).T
m_siteA,c_siteA = np.linalg.lstsq(A,y_siteA)[0]
      

print(f"slope = {m_siteA}, intercept = {c_siteA}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c_siteA),
                    r'$Slope = %.2f$' %(m_siteA),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x_siteA,y_siteA, marker='x', color='blue', label='Site A')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m_siteA + c_siteA,color='red',label='regression line site A')

plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize-3) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([0,50])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[ ]:


# plotting true pdff vs. predicted pdff


icc_result = compute_ICC(x_siteB,y_siteB)
print(f"iCC for site B: {icc_result['ICC'][1]}")
      
      
A = np.vstack([x_siteB,np.ones(x_siteB.shape[0])]).T
m_siteB,c_siteB = np.linalg.lstsq(A,y_siteB)[0]


#print(f"slope = {m}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$Intercept = %.2f$' %(c_siteB),
                    r'$Slope = %.2f$' %(m_siteB),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(x_siteB,y_siteB, marker='+', color='blue', label='site B')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m_siteB + c_siteB,color='red',label='regression line site B')
plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize-3) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([0,50])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[23]:


ylim = 10

index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


mean_val = (x_siteA + y_siteA)/2
diff_val = (x_siteA - y_siteA)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(mean_val, diff_val,marker='x', color='black')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

plt.ylim([-1*ylim,ylim])


# In[ ]:


mean_val = (x_siteB + y_siteB)/2
diff_val = (x_siteB - y_siteB)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(mean_val, diff_val,marker='x', color='black')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)

plt.ylim([-1*ylim,ylim])
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))


# In[ ]:


x = df_cross_val.r2star_liver_median_cnn[index]*r2star_max
y = df_cross_val.r2star_liver_median_true[index]*r2star_max

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


icc_result = compute_ICC(x_siteA,y_siteA)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x_siteA,np.ones(x_siteA.shape[0])]).T
m_siteA,c_siteA = np.linalg.lstsq(A,y_siteA)[0]
      

print(f"slope = {m_siteA}, intercept = {c_siteA}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c_siteA),
                    r'$Slope = %.2f$' %(m_siteA),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x_siteA,y_siteA, marker='x', color='blue', label='Site A')
# add a min and max 
x1 = np.append(x,[0,500],axis=0)
plt.plot(x1,x1*m_siteA + c_siteA,color='red',label='regression line site A')

plt.plot([0,500],[0,500],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize-3) # legend at upper left
#plt.xlabel('CNN-R2* (1/s)',fontsize=fontsize)
#plt.ylabel('Ref-6echo R2* (1/s)',fontsize=fontsize)
#plt.xlim([0,500])
#plt.ylim([0,500])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.6, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[ ]:


icc_result = compute_ICC(x_siteB,y_siteB)
print(f"iCC for site B: {icc_result['ICC'][1]}")
      
      
A = np.vstack([x_siteB,np.ones(x_siteB.shape[0])]).T
m_siteB,c_siteB = np.linalg.lstsq(A,y_siteB)[0]


#print(f"slope = {m}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$Intercept = %.2f$' %(c_siteB),
                    r'$Slope = %.2f$' %(m_siteB),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(x_siteB,y_siteB, marker='+', color='blue', label='site B')
# add a min and max 
x1 = np.append(x,[0,500],axis=0)
plt.plot(x1,x1*m_siteB + c_siteB,color='red',label='regression line site B')
plt.plot([0,500],[0,500],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize-3) # legend at upper left
#plt.xlabel('CNN-R2* (1/s)',fontsize=fontsize)
#plt.ylabel('Ref-6echo R2* (1/s)',fontsize=fontsize)
plt.xlim([0,500])
plt.ylim([0,500])
ax1.tick_params(axis='both',labelsize=13)
ax1.text(0.6, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[24]:


ylim = 250

index = df_cross_val.kfold_index == subset_index
x = df_cross_val.r2star_liver_median_true[index]*r2star_max
y = df_cross_val.r2star_liver_median_cnn[index]*r2star_max


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


mean_val = (x_siteA + y_siteA)/2
diff_val = (x_siteA - y_siteA)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(mean_val, diff_val,marker='x', color='black')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

plt.ylim([-1*ylim,ylim])


# In[ ]:


mean_val = (x_siteB + y_siteB)/2
diff_val = (x_siteB - y_siteB)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(mean_val, diff_val,marker='x', color='black')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)

plt.ylim([-1*ylim,ylim])
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))


# In[ ]:


index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


mean_val = (x_siteA + y_siteA)/2
diff_val = (x_siteA - y_siteA)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(mean_val, diff_val,marker='x', color='black')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
plt.xlabel('Average PDFF (%)',fontsize=16)
plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=13)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

plt.ylim([-1*ylim,ylim])


# In[ ]:





# ## Analysis between 2-point Dixon vs. CNN model C 

# In[135]:


index = df_cross_val.kfold_index == subset_index
# median hepatic values rather than mean
fontsize = 16
N_sub = 147

x = df_cross_val.pdff_liver_median_dixon[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100

### we will exclude outliers before we perform ICC 
dx_pdff = x - y
dx_pdff_mean = np.mean( dx_pdff )
dx_pdff_sd = np.std( dx_pdff )
x_no_outliers = x[  ( dx_pdff < (dx_pdff_mean + dx_pdff_sd*3) ) & ( dx_pdff > (dx_pdff_mean - dx_pdff_sd*3) )   ]
y_no_outliers = y[  ( dx_pdff < (dx_pdff_mean + dx_pdff_sd*3) ) & ( dx_pdff > (dx_pdff_mean - dx_pdff_sd*3) )   ]

print(f"number of data points after excluding outliders: {len(x_no_outliers)}")


# In[136]:


# # Dixon on FF on test A and test B

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


icc_result = compute_ICC(x_no_outliers,y_no_outliers)
print(f"iCC for both sites: {icc_result['ICC'][1]}")
      
# both sites
A = np.vstack([x_no_outliers,np.ones(x_no_outliers.shape[0])]).T
m,c = np.linalg.lstsq(A,y_no_outliers)[0]
      

print(f"slope = {m_siteA}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c),
                    r'$Slope = %.2f$' %(m),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x_siteA,y_siteA, marker='x', alpha= 0.5, color='blue', label='test A')
plt.scatter(x_siteB,y_siteB, marker='o', alpha=0.3, color='red', label='test B')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m + c,color='black',label='regression line')

plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([-60,60])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[125]:


# calculating diagnostic metric using PDFF > 5.5 as cutoff
pdff_cutoff = 5.5

TP = (y>=pdff_cutoff) & (x>=pdff_cutoff)
all_P = (y>=pdff_cutoff)
FP = (y < pdff_cutoff) & (x >= pdff_cutoff)

TN = (y< pdff_cutoff) & (x < pdff_cutoff)
all_N = (y < pdff_cutoff)
FN = (y >= pdff_cutoff) & (x < pdff_cutoff)


sen_dixon = np.sum(TP)/np.sum(all_P)
print("Dixon method diagnostic accuracy")
print(f"sensitivity = {sen_dixon}")
spec_dixon = np.sum(TN)/np.sum(all_N)
print(f"specificity = {spec_dixon}")


ppv_dixon = np.sum(TP)/(np.sum(TP) + np.sum(FP))
print(f"postive predictive value = {ppv_dixon}")
npv_dixon = np.sum(TN)/(np.sum(TN) + np.sum(FN) )
print(f"negative predictive value = {npv_dixon}")


# In[108]:


x = np.double([2,10,3])
y = np.double([5,14,10])
z = ( x < 6 ) & ( y > 6 )
print(z)


# In[26]:


# # Dixon on FF on both site TOGETHER

index = df_cross_val.kfold_index == subset_index
# median hepatic values rather than mean
fontsize = 16
N_sub = 147

x = df_cross_val.pdff_liver_median_dixon[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100


icc_result = compute_ICC(x,y)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
      

print(f"slope = {m_siteA}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c),
                    r'$Slope = %.2f$' %(m),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x,y, marker='x', alpha= 0.5, color='blue', label='paired hepatic PDFF values')
#plt.scatter(x_siteB,y_siteB, marker='o', alpha=0.3, color='red', label='test B')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m + c,color='black',label='regression line')

plt.plot([0,50],[0,50],'--',color='red',label='identity line')
plt.legend(loc=2, fontsize=fontsize) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([-60,60])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[117]:


# CNN inference on FF using model C on test A and test B
index = df_cross_val.kfold_index == 2
# median hepatic values rather than mean
fontsize = 16
x = df_cross_val.pdff_liver_median_cnn[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


icc_result = compute_ICC(x,y)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
      

print(f"slope = {m_siteA}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c),
                    r'$Slope = %.2f$' %(m),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x_siteA,y_siteA, marker='x', alpha= 0.5, color='blue', label='test A')
plt.scatter(x_siteB,y_siteB, marker='o', alpha=0.3, color='red', label='test B')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m + c,color='black',label='regression line')

plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.legend(loc=2, fontsize=fontsize) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([0,50])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[118]:


# calculating diagnostic metric using PDFF > 5.5 as cutoff
pdff_cutoff = 5.5 

TP = (y>=pdff_cutoff) & (x>=pdff_cutoff)
all_P = (y>=pdff_cutoff)
FP = (y < pdff_cutoff) & (x >= pdff_cutoff)

TN = (y< pdff_cutoff) & (x < pdff_cutoff)
all_N = (y < pdff_cutoff)
FN = (y >= pdff_cutoff) & (x < pdff_cutoff)


sen_dixon = np.sum(TP)/np.sum(all_P)
print("Dixon method diagnostic accuracy")
print(f"sensitivity = {sen_dixon}")
spec_dixon = np.sum(TN)/np.sum(all_N)
print(f"specificity = {spec_dixon}")


ppv_dixon = np.sum(TP)/(np.sum(TP) + np.sum(FP))
print(f"postive predictive value = {ppv_dixon}")
npv_dixon = np.sum(TN)/(np.sum(TN) + np.sum(FN) )
print(f"negative predictive value = {npv_dixon}")


# In[ ]:


# CNN inference on FF using model C TOGETHER
index = df_cross_val.kfold_index == 2
# median hepatic values rather than mean
fontsize = 16
x = df_cross_val.pdff_liver_median_cnn[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100


icc_result = compute_ICC(x,y)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
      

print(f"slope = {m_siteA}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c),
                    r'$Slope = %.2f$' %(m),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x,y, marker='x', alpha= 0.5, color='blue', label='paired hepatic PDFF values')
#plt.scatter(x_siteB,y_siteB, marker='o', alpha=0.3, color='red', label='test B')
# add a min and max 
x1 = np.append(x,[0,50],axis=0)
plt.plot(x1,x1*m + c,color='black',label='regression line')

plt.plot([0,50],[0,50],'--',color='red',label='identity line')
plt.legend(loc=2, fontsize=fontsize) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([0,50])
plt.ylim([0,50])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[56]:


# r2star inference usng model C on test A & test B

index = df_cross_val.kfold_index == 2
# median hepatic values rather than mean
fontsize = 16
x = df_cross_val.r2star_liver_median_cnn[index]*r2star_max
y = df_cross_val.r2star_liver_median_true[index]*r2star_max

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


icc_result = compute_ICC(x,y)
print(f"iCC for site A: {icc_result['ICC'][1]}")
      
# site A
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
      

print(f"slope = {m_siteA}, intercept = {c}")
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                   r'$Intercept = %.2f$' %(c),
                    r'$Slope = %.2f$' %(m),
                    r'$ICC = %.2f$' %(icc_result['ICC'][1]),
                    ))
fig, ax1 = plt.subplots(figsize=(8,8))
      
plt.scatter(x_siteA,y_siteA, marker='x', alpha= 0.5, color='blue', label='test A')
plt.scatter(x_siteB,y_siteB, marker='o', alpha=0.3, color='red', label='test B')
# add a min and max 
x1 = np.append(x,[0,500],axis=0)
plt.plot(x1,x1*m + c,color='black',label='regression line')

plt.plot([0,500],[0,500],'--',color='black',label='identity line')
plt.legend(loc=0, fontsize=fontsize) # legend at upper left
#plt.xlabel('CNN-FF (%)',fontsize=fontsize)
#plt.ylabel('Ref-6echo PDFF (%)',fontsize=fontsize)
plt.xlim([0,500])
plt.ylim([0,500])
ax1.tick_params(axis='both',labelsize=16)
#plt.legend(fontsize=fontsize,loc=0)
ax1.text(0.65, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize)


# In[ ]:


x = df_cross_val.pdff_liver_median_dixon[index]*100
y = df_cross_val.pdff_liver_median_true[index]*100

### we will exclude outliers before we perform ICC 
dx_pdff = x - y
dx_pdff_mean = np.mean( dx_pdff )
dx_pdff_sd = np.std( dx_pdff )
x_no_outliers = x[  ( dx_pdff < (dx_pdff_mean + dx_pdff_sd*3) ) & ( dx_pdff > (dx_pdff_mean - dx_pdff_sd*3) )   ]
y_no_outliers = y[  ( dx_pdff < (dx_pdff_mean + dx_pdff_sd*3) ) & ( dx_pdff > (dx_pdff_mean - dx_pdff_sd*3) )   ]


# In[52]:


# Bland-Altman plot of Dixon method on Site A and site B

index = df_cross_val.kfold_index == 2
x = df_cross_val.pdff_liver_median_true[index]*100
y= df_cross_val.pdff_liver_median_dixon[index]*100

# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


mean_val = (x_no_outliers + y_no_outliers)/2
diff_val = (x_no_outliers - y_no_outliers)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter((x_siteA + y_siteA)/2, (x_siteA - y_siteA),marker='x', alpha = 0.5, color='blue')
plt.scatter((x_siteB + y_siteB)/2, (x_siteB - y_siteB),marker='o', alpha = 0.3, color='red')
plt.axhline(md,           color='black', linestyle='--')
plt.axhline(md - 1.96*sd, color='black', linestyle='--')
plt.axhline(md + 1.96*sd, color='black', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])


# In[ ]:


# Bland-Altman plot of Dixon method on Site A and site B

index = df_cross_val.kfold_index == 2
x = df_cross_val.pdff_liver_median_true[index]*100
y= df_cross_val.pdff_liver_median_dixon[index]*100
z = df_cross_val.r2star_liver_median_true[index]*400

mean_val = (x + y)/2
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(z, (x - y),marker='x', alpha = 0.5, color='blue')
#plt.scatter((x_siteB + y_siteB)/2, (x_siteB - y_siteB),marker='o', alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='black', linestyle='--')
plt.axhline(md + 1.96*sd, color='black', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])


# In[ ]:


# Bland-Altman plot of CNN method on Site A and site B

index = df_cross_val.kfold_index == 2
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]


mean_val = (x + y)/2
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter((x_siteA + y_siteA)/2, (x_siteA - y_siteA),marker='x', alpha = 0.5, color='blue')
plt.scatter((x_siteB + y_siteB)/2, (x_siteB - y_siteB),marker='o', alpha = 0.3, color='red')
plt.axhline(md,           color='black', linestyle='--')
plt.axhline(md - 1.96*sd, color='black', linestyle='--')
plt.axhline(md + 1.96*sd, color='black', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])


# In[ ]:


# Bland-Altman plot of CNN method on Site A and site B

index = df_cross_val.kfold_index == 2
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100
z = df_cross_val.r2star_liver_median_true[index]*400

mean_val = (x + y)/2
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(z, (x - y),marker='x', alpha = 0.5, color='blue')
#plt.scatter((x_siteB + y_siteB)/2, (x_siteB - y_siteB),marker='o', alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='black', linestyle='--')
plt.axhline(md + 1.96*sd, color='black', linestyle='--')
#plt.xlabel('Average PDFF (%)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN-Predicted PDFF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
ax1.text(0.05, 0.05, textstr, transform = ax1.transAxes, fontsize=fontsize, 
         color = 'black',
         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])


# ## Examine the effect of site, reference PDFF, R2* on the accuracy of PDFF prediction 

# In[160]:


# reference PDFF vs. difference of PDFF 

index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_dixon[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]

dx_siteA = x_siteA - y_siteA
dx_siteB = x_siteB - y_siteB

z = df_cross_val.r2star_liver_median_true[index]
z[test_site_index == 1] = z[test_site_index == 1]
z = z*r2star_max
z_siteA = z[test_site_index == 1]
z_siteB = z[test_site_index == 2]


x_siteA_valid = x_siteA[y_siteA >=0]
dx_siteA_valid = dx_siteA[y_siteA >=0]
z_siteA_valid = z_siteA[y_siteA >=0]
dx_siteB_valid = dx_siteB[y_siteB >=0]
x_siteB_valid = x_siteB[y_siteB >=0]
z_siteB_valid = z_siteB[y_siteB >=0]
dx_siteA_invalid = dx_siteA[y_siteA <0]
x_siteA_invalid = x_siteA[y_siteA <0]
z_siteA_invalid = z_siteA[y_siteA < 0]
dx_siteB_invalid = dx_siteB[y_siteB <0]
x_siteB_invalid = x_siteB[y_siteB <0]
z_siteB_invalid = z_siteB[y_siteB < 0]

markersize1=100
markersize2=80

mean_val = z
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(x_siteA_valid, dx_siteA_valid, marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(x_siteA_invalid, dx_siteA_invalid,marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(x_siteA_invalid, dx_siteA_invalid,marker='o',s= markersize1, alpha = 0.3, color='blue')
plt.scatter(x_siteB_valid, dx_siteB_valid,marker='D', s= markersize2, alpha = 0.5, color='red', facecolor="None")
plt.scatter(x_siteB_invalid, dx_siteB_invalid,marker='D', s=markersize2, alpha = 0.5, color='red',facecolor="None")
plt.scatter(x_siteB_invalid, dx_siteB_invalid,marker='D',s= markersize2, alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('6-echo R2*(1/2)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - 2-pt Dixon FF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
#ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
#         color = 'black',
#         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])


# In[200]:


index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_dixon[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]

dx_siteA = x_siteA - y_siteA
dx_siteB = x_siteB - y_siteB

z = df_cross_val.r2star_liver_median_true[index]
z[test_site_index == 1] = z[test_site_index == 1]
z = z*r2star_max
z_siteA = z[test_site_index == 1]
z_siteB = z[test_site_index == 2]

dx_siteA_valid = dx_siteA[y_siteA >=0]
z_siteA_valid = z_siteA[y_siteA >=0]
dx_siteB_valid = dx_siteB[y_siteB >=0]
z_siteB_valid = z_siteB[y_siteB >=0]
dx_siteA_invalid = dx_siteA[y_siteA <0]
z_siteA_invalid = z_siteA[y_siteA < 0]
dx_siteB_invalid = dx_siteB[y_siteB <0]
z_siteB_invalid = z_siteB[y_siteB < 0]



mean_val = z
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(z_siteA_valid, dx_siteA_valid, marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(z_siteA_invalid, dx_siteA_invalid,marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(z_siteA_invalid, dx_siteA_invalid,marker='o',s= markersize1, alpha = 0.3, color='blue')
plt.scatter(z_siteB_valid, dx_siteB_valid,marker='D', s= markersize2, alpha = 0.5, color='red', facecolor="None")
plt.scatter(z_siteB_invalid, dx_siteB_invalid,marker='D', s=markersize2, alpha = 0.5, color='red',facecolor="None")
plt.scatter(z_siteB_invalid, dx_siteB_invalid,marker='D',s= markersize2, alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('6-echo R2*(1/2)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - 2-pt Dixon FF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
#ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
#         color = 'black',
#         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 65
plt.ylim([-1*ylim,ylim])
#plt.xlim([20,100])


# In[203]:


# scatter plot with color coded for R2*
markersize1=30
fig, ax1 = plt.subplots(figsize=(18,8))
jitter_esp = (np.random.rand(len(x))-0.5)*0.5
x_jitter = x + jitter_esp
plt.scatter(y, x_jitter, marker='o',s=markersize1, c=np.log10(z), cmap='RdBu')
plt.axvline(0, color='blue', linestyle='--')
plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.colorbar()
plt.xlim([-60,45])
plt.ylim([0,45])
ax1.tick_params(axis='both',labelsize=16)


# In[ ]:


# multivariabe analysis with indepdendent variable site, reference PDFF, and reference R2*
x = df_cross_val.pdff_liver_median_true[index]*100
y1 = df_cross_val.pdff_liver_median_dixon[index]*100
y2 = df_cross_val.pdff_liver_median_cnn[index]*100


x0 =  (2- test_site_index)*1.5
x1 = df_cross_val.pdff_liver_median_true[index]*100
x2 = df_cross_val.r2star_liver_median_true[index]*r2star_max
dy1 = x-y1
dy2 = x-y2


#X = np.transpose( np.vstack([x0,x1,x2]),[1,0])

#X2 = sm.add_constant(X)
#est = sm.OLS(y, X2)
#est2 = est.fit()
#print(est2.summary())


# In[ ]:


# multivariabe analysis with indepdendent variable site, reference PDFF, and reference R2*
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_dixon[index]*100


x0 = np.array( test_site_index - 1, dtype = np.int32 )
tmp1 = df_cross_val.pdff_liver_median_true[index]*100
x1 = tmp1
tmp2 = df_cross_val.r2star_liver_median_true[index]*r2star_max
x2 = tmp2
tmp3 = x - y 
y = tmp3
X = np.transpose( np.vstack([x0,x1,x2]),[1,0])
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:


print(mlr.intercept_)
print(mlr.coef_)


# In[ ]:


# assemble the table 
tmp_df = {'field_strength':x0,
       'ref_pdff': x1,
       'ref_r2star': x2,
       'dpdff_dixon':dy1,
        'dpdff_cnn':dy2
         }

df_2pt_dixon = pd.DataFrame(data=tmp_df)
df_2pt_dixon.head()
df_2pt_dixon.to_csv('/data/multivariate_analysis_dpdff.csv')


# In[ ]:


plt.scatter(x2, y)


# In[162]:


# reference PDFF vs. difference of PDFF 

index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]

dx_siteA = x_siteA - y_siteA
dx_siteB = x_siteB - y_siteB

z = df_cross_val.r2star_liver_median_true[index]
z[test_site_index == 1] = z[test_site_index == 1]
z = z*r2star_max
z_siteA = z[test_site_index == 1]
z_siteB = z[test_site_index == 2]


x_siteA_valid = x_siteA[y_siteA >=0]
dx_siteA_valid = dx_siteA[y_siteA >=0]
z_siteA_valid = z_siteA[y_siteA >=0]
dx_siteB_valid = dx_siteB[y_siteB >=0]
x_siteB_valid = x_siteB[y_siteB >=0]
z_siteB_valid = z_siteB[y_siteB >=0]
dx_siteA_invalid = dx_siteA[y_siteA <0]
x_siteA_invalid = x_siteA[y_siteA <0]
z_siteA_invalid = z_siteA[y_siteA < 0]
dx_siteB_invalid = dx_siteB[y_siteB <0]
x_siteB_invalid = x_siteB[y_siteB <0]
z_siteB_invalid = z_siteB[y_siteB < 0]



mean_val = z
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(x_siteA_valid, dx_siteA_valid, marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(x_siteA_invalid, dx_siteA_invalid,marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(x_siteA_invalid, dx_siteA_invalid,marker='o',s= markersize1, alpha = 0.3, color='blue')
plt.scatter(x_siteB_valid, dx_siteB_valid,marker='D', s= markersize2, alpha = 0.5, color='red', facecolor="None")
plt.scatter(x_siteB_invalid, dx_siteB_invalid,marker='D', s=markersize2, alpha = 0.5, color='red',facecolor="None")
plt.scatter(x_siteB_invalid, dx_siteB_invalid,marker='D',s= markersize2, alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('6-echo R2*(1/2)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - 2-pt Dixon FF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
#ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
#         color = 'black',
#         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 60
plt.ylim([-1*ylim,ylim])


# In[204]:


index = df_cross_val.kfold_index == subset_index
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100


# we will plot site A and site B separately 
x_siteA = x[test_site_index == 1]
y_siteA = y[test_site_index == 1]
x_siteB = x[test_site_index == 2]
y_siteB = y[test_site_index == 2]

dx_siteA = x_siteA - y_siteA
dx_siteB = x_siteB - y_siteB

z = df_cross_val.r2star_liver_median_true[index]
z[test_site_index == 1] = z[test_site_index == 1]
z = z*r2star_max
z_siteA = z[test_site_index == 1]
z_siteB = z[test_site_index == 2]

dx_siteA_valid = dx_siteA[y_siteA >=0]
z_siteA_valid = z_siteA[y_siteA >=0]
dx_siteB_valid = dx_siteB[y_siteB >=0]
z_siteB_valid = z_siteB[y_siteB >=0]
dx_siteA_invalid = dx_siteA[y_siteA <0]
z_siteA_invalid = z_siteA[y_siteA < 0]
dx_siteB_invalid = dx_siteB[y_siteB <0]
z_siteB_invalid = z_siteB[y_siteB < 0]



mean_val = z
diff_val = (x - y)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(8,8))
plt.scatter(z_siteA_valid, dx_siteA_valid, marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(z_siteA_invalid, dx_siteA_invalid,marker='o',s=markersize1, alpha = 0.5, color='blue', facecolor="None")
plt.scatter(z_siteA_invalid, dx_siteA_invalid,marker='o',s= markersize1, alpha = 0.3, color='blue')
plt.scatter(z_siteB_valid, dx_siteB_valid,marker='D', s= markersize2, alpha = 0.5, color='red', facecolor="None")
plt.scatter(z_siteB_invalid, dx_siteB_invalid,marker='D', s=markersize2, alpha = 0.5, color='red',facecolor="None")
plt.scatter(z_siteB_invalid, dx_siteB_invalid,marker='D',s= markersize2, alpha = 0.3, color='red')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
#plt.xlabel('6-echo R2* (1/s)',fontsize=16)
#plt.ylabel('Ref-6echo PDFF - CNN FF (%) ',fontsize=16)
#plt.title('Reference vs. CNN-predicted hepatic PDFF',fontsize=16)

print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")

ax1.tick_params(axis='both',labelsize=16)
# create a text string within the plot to display regression and correlation
textstr = '\n'.join((
                    r'$bias = %.2f$' %(md),
                    r'$Limit\ of\ agreement = (%.2f, %.2f)$' %(md - 1.96*sd,md + 1.96*sd)
                    ))
#ax1.text(0.05, 0.88, textstr, transform = ax1.transAxes, fontsize=fontsize, 
#         color = 'black',
#         bbox=dict(facecolor=None, fc='white', edgecolor='black',boxstyle='square,pad=0.5'))

ylim = 60
plt.ylim([-1*ylim,ylim])


# In[206]:


# scatter plot with color coded for R2*
markersize1=50
fig, ax1 = plt.subplots(figsize=(18,8))
jitter_esp = (np.random.rand(len(x))-0.5)*0.5
x_jitter = x + jitter_esp
plt.scatter(y, x_jitter, marker='o',s=markersize1, c=np.log10(z), cmap='RdBu')
plt.axvline(0, color='blue', linestyle='--')
plt.plot([0,50],[0,50],'--',color='black',label='identity line')
plt.colorbar()
#plt.xlim([0,45])
#plt.ylim([0,45])
plt.xlim([-60,45])
plt.ylim([0,45])
ax1.tick_params(axis='both',labelsize=16)


# In[ ]:


# multivariabe analysis with indepdendent variable site, reference PDFF, and reference R2*
x = df_cross_val.pdff_liver_median_true[index]*100
y = df_cross_val.pdff_liver_median_cnn[index]*100
x0 = np.array( test_site_index - 1, dtype = np.int32 )
tmp1 = df_cross_val.pdff_liver_median_true[index]*100
x1 = tmp1
tmp2 = df_cross_val.r2star_liver_median_true[index]*r2star_max
x2 = tmp2
tmp3 = x - y 
y = tmp3
X = np.transpose( np.vstack([x0,x1,x2]),[1,0])
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:


print(mlr.intercept_)
mlr.score()


# In[ ]:





# In[ ]:


index = df_cross_val.kfold_index == subset_index
x1_raw = df_cross_val.pdff_liver_median_true[index]*100
x2_raw = df_cross_val.pdff_liver_median_dixon[index]*100

y1_raw = df_cross_val.r2star_liver_median_true[index]*r2star_max
y1_raw[test_site_index == 1] = y1_raw[test_site_index == 1]*2


index1 = np.where( df_cross_val.pdff_liver_median_dixon[index] >= 0 )
index2 = np.where( df_cross_val.pdff_liver_median_dixon[index] < 0 )
#print(np.where( np.abs(df_cross_val.pdff_liver_mean_dixon - df_cross_val.pdff_liver_mean_true) > 0.2 ))
#print(len(index[0]))
x1_valid = x1_raw[index1[0]]
x2_valid = x2_raw[index1[0]]

x1_outlier = x1_raw[index2[0]]
x2_outlier = x2_raw[index2[0]]


mean_val = y1_raw[index1[0]]
diff_val = (x1_valid - x2_valid)
md        = np.mean(diff_val)       # Mean of the difference
sd        = np.std(diff_val, axis=0)    # Standard deviation of the difference
fig, ax1 = plt.subplots(figsize=(10,8))
plt.scatter(y1_raw[index1[0]], 
            x1_valid - x2_valid,marker='x', color='black',alpha=0.5, label='2pt-Dixon FF>0.0')
plt.scatter(y1_raw[index2[0]],
            x1_outlier-x2_outlier, marker='o', color='red', alpha=0.3, label='2pt-Dixon FF<0.0')
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
plt.xlabel('hepatic R2* (1/s)',fontsize=16)
plt.ylabel('6-echo PDFF - 2pt-Dixon FF (%) ',fontsize=16)
plt.title('Effect of R2* on 2pt-Dixon FF accuracy',fontsize=16)
#plt.ylim([-15,15])
print(f"bias = {md}, LoA = ({md - 1.96*sd},{md + 1.96*sd})")
plt.legend(fontsize=16,loc=2)


ax1.tick_params(axis='both',labelsize=13)


# ## Look at individual examples

# In[ ]:


print(r2star_liver_median_6echo_OKA)


# In[ ]:


index_r2 = np.where(r2star_liver_median_true > 200)


# In[ ]:


case_lst_OKA[133]


# In[ ]:


case_ii = 34
test_cases = get_X_Y(hd5_fn,test_lst[case_ii:case_ii+1], 
                       liver_area_th=liver_area_th,
                       r2star_max = r2star_max,
                       img_normalization='batch',
                       training_flag=False)


# In[ ]:


case_index = 0
input_images = test_cases['input_data']
Pred = eval_model_batch(net,input_images,batch_size)

test_case_X = test_cases['input_data'][ test_cases['case_indices'] == case_index,:,:,: ]
test_case_Y = test_cases['output_data'][ test_cases['case_indices'] == case_index,:,:,: ]
test_case_Pred = Pred[ test_cases['case_indices'] == case_index,:,:,: ] 
lava_segmask = test_cases['lava_segmask'][ :,:, test_cases['case_indices'] == case_index]
idealiq_segmask = test_cases['idealiq_segmask'][ :,:, test_cases['case_indices'] == case_index ]
lava_segmask = idealiq_segmask

case_data['input_data'] = test_case_X
case_data['output_data'] = test_case_Y
case_data['pred_data'] = test_case_Pred
case_data['lava_segmask'] = lava_segmask
case_data['idealiq_segmask'] = idealiq_segmask


# computing various parametric maps
pdffmap_cnn_pred = np.transpose(test_case_Pred[:,:,:,0],[1,2,0])
r2starmap_cnn_pred = np.transpose(test_case_Pred[:,:,:,1],[1,2,0])
pdffmap_test = np.transpose(test_case_Y[:,:,:,0],[1,2,0])
r2starmap_test = np.transpose(test_case_Y[:,:,:,1],[1,2,0])
ip_test = np.transpose(test_case_X[:,:,:,0],[1,2,0])
op_test = np.transpose(test_case_X[:,:,:,1],[1,2,0])


# In[ ]:


case_ii = 34

print(df_cross_val.r2star_liver_median_true[case_ii]*400)
print(df_cross_val.pdff_liver_median_true[case_ii]*100)

print(df_cross_val.r2star_liver_median_cnn[case_ii]*400)
print(df_cross_val.pdff_liver_median_cnn[case_ii]*100)
print(df_cross_val.pdff_liver_median_dixon[case_ii]*100)


# In[ ]:


for ii in range(0,25,2):
    imshow(test_cases['input_data'][ii,:,:,0], pdffmap_cnn_pred[:,:,ii], r2starmap_cnn_pred[:,:,ii], test_cases['output_data'][ii,:,:,1])


# In[ ]:


ii = 10
fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(test_cases['input_data'][ii,:,:,1],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(pdffmap_cnn_pred[:,:,ii],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(pdffmap_test[:,:,ii],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(r2starmap_cnn_pred[:,:,ii],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(r2starmap_test[:,:,ii],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


ff_dixon = (ip_test - op_test)/(ip_test)/2
fig, ax1 = plt.subplots(figsize=(8,8))
plt.imshow(ff_dixon[:,:,ii],cmap='gray',clim=[0,1])
plt.axis('off')


# In[ ]:


print(tmp2)


# In[ ]:



x = df_cross_val.pdff_liver_median_true*100
y = df_cross_val.pdff_liver_median_dixon*100
z = df_cross_val.r2star_liver_median_true*r2star_max
index_r2 = np.where(z > 200)
print(index_r2)


# In[ ]:


print( np.where( (x > 30)  &  (abs(x-y) >5) )  )


# In[ ]:


z[index_r2[0]]


# In[ ]:


75*2


# In[ ]:




