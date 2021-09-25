'''
This is the implementation of the band influence algorithm
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
from skimage import io
import time
from datetime import datetime
import scipy.io as sio
from scipy import integrate
from scipy.signal import argrelextrema

import torch
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score

from methods import train_cnn
from dataset import Pnt_eb_Dataset, lines_aug,load_data,load_data2


'''
traditional classification model
Band influence algorithm (using dt,knn,svm)
'''
def remove_equal_value2(acc, times=50):
    for _times in range(times):
        ###### Remove the equal adjacent values, from left to right#######
        acc_cp = acc.copy()
        for i in range(1,len(acc)-1):
            acc_i = acc[i]
            if acc_i == acc[i+1]:
                acc_cp[i] = (acc[i-1]+acc[i+1])/2
        ##### Remove the equal adjacent values, from right to left########
        acc_cp_reverse = acc_cp.copy()[::-1]
        acc_cp2 = acc_cp.copy()[::-1]
        for i in range(1,len(acc_cp2)-1):
            acc_i = acc_cp_reverse[i]
            if acc_i == acc_cp_reverse[i+1]:
                acc_cp2[i] = (acc_cp_reverse[i-1]+acc_cp_reverse[i+1])/2
        acc = acc_cp2[::-1]
    return acc

def Bia_method(sel_bands_num,spectral_train, label_train,func_eb,Step=3):
    # 
    acc_or = np.ones((128))
    for ii in range(0,128):
        img_eb = spectral_train.copy()
        if Step == 1:
            img_eb[:,[ii]] = 0
            y_predict = func_eb.predict(img_eb)
            svm_acc_i = accuracy_score(label_train, y_predict)
            acc_or[ii] = svm_acc_i
        elif Step == 3 and ii < 128-2:
            img_eb[:,[ii,ii+1,ii+2]] = 0
            y_predict = func_eb.predict(img_eb)
            svm_acc_i = accuracy_score(label_train, y_predict)
            acc_or[ii+1] = svm_acc_i
        elif Step == 5 and ii < 128-4:
            img_eb[:,[ii,ii+1,ii+2,ii+3,ii+4,]] = 0
            y_predict = func_eb.predict(img_eb)
            svm_acc_i = accuracy_score(label_train, y_predict)
            acc_or[ii+2] = svm_acc_i
    acc_smooth = remove_equal_value2(acc_or)
    
    # 
    spectral_mean = []
    for class_i in range(1,4):
        spectral_i = spectral_train[label_train==class_i]
        spectral_i_mean = np.mean(spectral_i, axis=0)
        spectral_mean.append(spectral_i_mean)
    spectral_mean = np.array(spectral_mean)
    # generate SD
    cha_all = np.zeros((spectral_mean.shape[1]))
    for i in range(spectral_mean.shape[0]-1):
        b = i
        while b < (spectral_mean.shape[0]-1):
            cha = np.abs(spectral_mean[i] - spectral_mean[b+1])
            cha_all = cha_all + cha
            b += 1
    # print(cha_all)
    integral_all = integrate.trapz(cha_all,)

    # 
    part = 3
    distance = len(cha_all)//part
    selected_bands = np.array([],dtype=np.uint32)
    rest = sel_bands_num
    start_band = 0
    for part_i in range(part):
        # local maximum
        maxInd = argrelextrema(acc_smooth, np.greater) # Return the index value
        end_band = distance*(part_i+1)
        if end_band not in maxInd[0]:
            end_band_idx = np.argmin(np.abs(maxInd[0] - end_band)) # 
            end_band = maxInd[0][end_band_idx]
        # The value of the start band cannot be greater than the value of the end band.
        if end_band < distance*(part_i+1)-distance//2 or end_band > distance*(part_i+1)+distance//2:
            end_band = distance*(part_i+1)
        if start_band > end_band or start_band == end_band:
            raise ValueError
        
        # 
        # Calculate the integral of each part and the corresponding number of feature bands
        integral01 = integrate.trapz(cha_all[start_band:end_band],)/integral_all
        if part_i == (part-1):
            integral01 = integrate.trapz(cha_all[start_band : ],)/integral_all
        band_num_i = np.round(integral01*sel_bands_num).astype(np.uint32)

        # Extract part of the ACC curve
        acc_part = acc_smooth[start_band : end_band]
        if part_i == (part-1):
            band_num_i = rest
            acc_part = acc_smooth[start_band : ]
        
        #  extract local min  ## np.less,np.greater 
        minInd = argrelextrema(acc_part, np.less) # 
        index_value = np.array(minInd, dtype=np.uint32)+(start_band)
        rest = rest - band_num_i
        start_band = end_band
        value01 = acc_smooth[index_value]
        
        #  If minimum ACC is greater than 85%, the Step size is set to 5.
        region_min= np.min(acc_part)
        if region_min > 0.85 and Step==3:
            raise ValueError

        if index_value.ndim == 1:
            index_value = np.expand_dims(index_value,axis=0)
        if value01.ndim == 1:
            value01 = np.expand_dims(np.array(value01),axis=0)
        idx_value = np.concatenate((index_value, value01), axis=0)
        #
        idx_sort = idx_value.T[np.lexsort(idx_value)].T
        
        # 
        # When the number of local extremes is less than the number of characteristic bands in the partition, 
        # the feature bands are selected from small to large with a Step size of 2.
        while idx_sort.shape[1] < band_num_i:
            idx02 = np.arange(0,distance,2) + (part_i*distance)
            value02 = acc_smooth[idx02]
            idx_value02=np.concatenate((np.expand_dims(idx02,axis=0), np.expand_dims(np.array(value02),axis=0)), axis=0)
            idx_sort2 = idx_value02.T[np.lexsort(idx_value02)].T
            idx_sort = np.concatenate((idx_sort, idx_sort2), axis=1)
        out_idx_all = np.array(idx_sort[0,:], dtype=np.int32)
        out_idx_part=[]
        for i in out_idx_all:
            if i not in out_idx_part:
                out_idx_part.append(i)
        out_idx_part = out_idx_part[:band_num_i]
        
        # 
        selected_bands = np.append(selected_bands,np.array(out_idx_part,dtype=np.uint32))
    print('bia selected bands',len(selected_bands),selected_bands)
    return selected_bands,acc_smooth


'''
CNN classification model
Band influence algorithm (using ShuffleNet V2)
'''
def read_mat(path):
    load_mat = sio.loadmat(path)
    img = load_mat['image']
    return img

def gen_acc_curve(args, net, lines_train, input_shape,Step=3):
    acc_or = np.ones((128))
    for ii in range(0,128-2):
        if Step == 3 and ii < 128-2:
            eb=[ii,ii+1,ii+2]
        elif Step == 5 and ii < 128-4:
            eb=[ii,ii+1,ii+2,ii+3,ii+4,]
        eb_dataset = Pnt_eb_Dataset(lines_train, input_shape, eb=eb, train=False)
        eb_loader = DataLoader(eb_dataset, shuffle=False, batch_size=32, num_workers=2, 
                                pin_memory=True,drop_last=False)
        result_all = np.array([])
        label_all = np.array([])
        with torch.no_grad():
            for n_iter, (image, label) in enumerate(eb_loader):
                if args.gpu:
                    image = image.cuda()
                    label = label.cuda()
                output = net(image)
                result = torch.max(output, -1)[1]# 返回最大值的索引
                result_batch = result.cpu().numpy()
                result_all = np.append(result_all,result_batch)
                label_batch = label.cpu().numpy()
                label_all = np.append(label_all,label_batch)
        label_all = label_all.astype(np.int8)
        result_all = result_all.astype(np.int8)
        acc_i = accuracy_score(label_all, result_all)

        if Step == 3 and ii < 128-2:
            acc_or[ii+1] = acc_i
        elif Step == 5 and ii < 128-4:
            acc_or[ii+2] = acc_i
    return acc_or


def cal_mean_spectral(lines_train):
    train_data = []
    train_label = []
    for line in lines_train:
        line=line.replace('[', '')
        line=line.replace(']', '')
        line=line.replace(',', '')
        mat_path = line.split(' ')[0]
        data_i = read_mat(mat_path)
        mask_path = mat_path.replace('image', 'label')
        mask_path = mask_path.replace('mat', 'png')
        mask_i = io.imread(mask_path,as_gray=True)
        kernel_pixel = data_i[mask_i != 0]
        kernel_mean = np.mean(kernel_pixel, axis=0) 
        label_i = int(line.split(' ')[1][0])
        train_data.append(kernel_mean)
        train_label.append(label_i)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    spectral_mean = []
    for class_i in range(1,4):
        spectral_i = train_data[train_label==class_i]
        spectral_i_mean = np.mean(spectral_i, axis=0)
        spectral_mean.append(spectral_i_mean)
    spectral_mean = np.array(spectral_mean)
    return spectral_mean


def check_Step(sel_bands_num, acc_or,spectral_mean,):
    # remove equal values
    acc_smooth = remove_equal_value2(acc_or)
    # generate SD
    cha_all = np.zeros((spectral_mean.shape[1]))
    for i in range(spectral_mean.shape[0]-1):
        b = i
        while b < (spectral_mean.shape[0]-1):
            cha = np.abs(spectral_mean[i] - spectral_mean[b+1])
            cha_all = cha_all + cha
            b += 1
    integral_all = integrate.trapz(cha_all,)
    
    # 
    part = 3
    distance = len(cha_all)//part
    rest = sel_bands_num
    start_band = 0
    for part_i in range(part):
        # local maximum
        maxInd = argrelextrema(acc_smooth, np.greater)
        end_band = distance*(part_i+1)
        if end_band not in maxInd[0]:
            end_band_idx = np.argmin(np.abs(maxInd[0] - end_band))  # 
            end_band = maxInd[0][end_band_idx]
        # The value of the start band cannot be greater than the value of the end band.
        if end_band < distance*(part_i+1)-distance//2 or end_band > distance*(part_i+1)+distance//2:
            end_band = distance*(part_i+1)
        if start_band > end_band or start_band == end_band:
            raise ValueError
        
        # 
        # Calculate the integral of each part and the corresponding number of feature bands
        integral01 = integrate.trapz(cha_all[start_band:end_band],)/integral_all
        if part_i == (part-1):
            integral01 = integrate.trapz(cha_all[start_band : ],)/integral_all
        band_num_i = np.round(integral01*sel_bands_num).astype(np.uint32)

        # Extract part of the ACC curve
        acc_part = acc_smooth[start_band : end_band]
        if part_i == (part-1):
            band_num_i = rest
            acc_part = acc_smooth[start_band : ]
        
        #  extract local min  ## np.less,np.greater 
        minInd = argrelextrema(acc_part, np.less)
        index_value = np.array(minInd, dtype=np.uint32)+(start_band)
        rest = rest - band_num_i
        start_band = end_band
        value01 = acc_smooth[index_value]
        
        #
        region_min= np.min(value01)
        if region_min > 0.85:
            raise ValueError
    return


def Bia_method_cnn(sel_bands_num, acc_or,spectral_mean,):
    # smooth equal values
    acc_smooth = remove_equal_value2(acc_or)
    # generate SD
    cha_all = np.zeros((spectral_mean.shape[1]))
    for i in range(spectral_mean.shape[0]-1):
        b = i
        while b < (spectral_mean.shape[0]-1):
            cha = np.abs(spectral_mean[i] - spectral_mean[b+1])
            cha_all = cha_all + cha
            b += 1
    # print(cha_all)
    integral_all = integrate.trapz(cha_all,)
    
    #
    part = 3
    distance = len(cha_all)//part
    selected_bands = np.array([],dtype=np.uint32)
    rest = sel_bands_num
    start_band = 0
    for part_i in range(part):
        # local maximum
        maxInd = argrelextrema(acc_smooth, np.greater)
        end_band = distance*(part_i+1)
        if end_band not in maxInd[0]:
            end_band_idx = np.argmin(np.abs(maxInd[0] - end_band)) # 
            end_band = maxInd[0][end_band_idx]
        # The value of the start band cannot be greater than the value of the end band.
        if end_band < distance*(part_i+1)-distance//2 or end_band > distance*(part_i+1)+distance//2:
            end_band = distance*(part_i+1)
        if start_band > end_band or start_band == end_band:
            raise ValueError
        
        # 
        # Calculate the integral of each part and the corresponding number of feature bands
        integral01 = integrate.trapz(cha_all[start_band:end_band],)/integral_all
        if part_i == (part-1):
            integral01 = integrate.trapz(cha_all[start_band : ],)/integral_all
        band_num_i = np.round(integral01*sel_bands_num).astype(np.uint32)
        
        # Extract part of the ACC curve
        acc_part = acc_smooth[start_band : end_band]
        if part_i == (part-1):
            band_num_i = rest
            acc_part = acc_smooth[start_band : ]
        
        #  extract local min  ## np.less,np.greater 
        minInd = argrelextrema(acc_part, np.less)
        index_value = np.array(minInd, dtype=np.uint32)+(start_band)
        rest = rest - band_num_i
        start_band = end_band
        value01 = acc_smooth[index_value]
        if index_value.ndim == 1:
            index_value = np.expand_dims(index_value,axis=0)
        if value01.ndim == 1:
            value01 = np.expand_dims(np.array(value01),axis=0)
        idx_value = np.concatenate((index_value, value01), axis=0)
        #
        idx_sort = idx_value.T[np.lexsort(idx_value)].T
        
        # 
        # When the number of local extremes is less than the number of characteristic bands in the partition, 
        # the feature bands are selected from small to large with a Step size of 2.
        while idx_sort.shape[1] < band_num_i:
            idx02 = np.arange(0,distance,2) + (part_i*distance)
            value02 = acc_smooth[idx02]
            idx_value02=np.concatenate((np.expand_dims(idx02,axis=0), np.expand_dims(np.array(value02),axis=0)), axis=0)
            idx_sort2 = idx_value02.T[np.lexsort(idx_value02)].T
            idx_sort = np.concatenate((idx_sort, idx_sort2), axis=1)
        out_idx_all = np.array(idx_sort[0,:], dtype=np.int32)
        out_idx_part=[]
        for i in out_idx_all:
            if i not in out_idx_part:
                out_idx_part.append(i)
        out_idx_part = out_idx_part[:band_num_i]
        # 
        selected_bands = np.append(selected_bands,np.array(out_idx_part,dtype=np.uint32))
    print('bia selected bands',len(selected_bands),selected_bands)
    return selected_bands


def dt_val(img_train, gt_train,img_test,):
    dt_func = DT(criterion="entropy").fit(img_train, gt_train)
    y_predict = dt_func.predict(img_test)
    return y_predict, dt_func

def knn_val(img_train, gt_train,img_test,):
    knn_func = KNN(n_neighbors=5).fit(img_train, gt_train)
    y_predict = knn_func.predict(img_test)
    return y_predict, knn_func

# Default kernel is 'rbf'
def svm_val_128(img_train, gt_train,img_test,):
    # Determine the best hyperparameters of svm
    params = [{'C': np.logspace(0 ,9 ,11 ,base =1.8), 'gamma':np.logspace(0 ,9 ,11 ,base =1.8),}]
    model = model_selection.GridSearchCV(SVC(), params, n_jobs=4,cv=5,return_train_score=True,)
    model.fit(img_train, gt_train)
    svm_func = model.best_estimator_
    y_predict = svm_func.predict(img_test)
    return y_predict,svm_func

def svm_val(img_train, gt_train,img_test,):
    # Determine the best hyperparameters of svm
    params = [{'C': np.logspace(0 ,9 ,11 ,base =1.8), 'gamma':np.logspace(0 ,9 ,11 ,base =1.8),}]
    model = model_selection.GridSearchCV(SVC(), params, n_jobs=4,cv=5,return_train_score=True,)
    model.fit(img_train, gt_train)
    svm_func = model.best_estimator_
    y_predict = svm_func.predict(img_test)
    return y_predict



######################################
# --use dt_knn_svm as classifier----##
######################################
def dt_knn_svm(val_method,bs_method,sel_bands_num,times,excel_name,ratio):
    # load all data
    spectral_train,label_train,spectral_test,label_test = load_data(times,excel_name,ratio)
    img_train = spectral_train
    gt_train = label_train
    img_test = spectral_test
    gt_test = label_test
    # generate all bands acc, and trained classifier
    if val_method == 'svm':
        y_128_predict, func = svm_val_128(img_train, gt_train,img_test,gt_test,
                                in_channels=128,val_method=val_method,method_i=bs_method)
    elif val_method == 'knn':
        y_128_predict, func = knn_val(img_train, gt_train,img_test,gt_test,in_channels=128, 
                                val_method=val_method,method_i=bs_method)
    elif val_method == 'dt':
        y_128_predict, func = dt_val(img_train, gt_train,img_test,gt_test,in_channels=128, 
                                val_method=val_method,method_i=bs_method)

    ### extract feature bands
    selected_bands = []
    for bands_num in range(4, sel_bands_num+1,2):
        try:
            selected_band_i,acc_smooth = Bia_method(bands_num,spectral_train, label_train,func_eb=func,Step=3)
        except :
            print('Warning: use Step=5.')
            selected_band_i,acc_smooth = Bia_method(bands_num,spectral_train, label_train,func_eb=func,Step=5)
        selected_band_i = list(np.sort(selected_band_i))
        selected_bands.append(selected_band_i)
    return selected_bands




class Parameters:
    def __init__(self,):
        super(Parameters, self).__init__()
        self.gpu = True
        self.batch_size = 32
        self.warm = 1
        self.resume = False
        self.class_num = 3
        self.CHECKPOINT_PATH = 'checkpoint'
        self.EPOCH = 50
        self.DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        self.TIME_NOW = datetime.now().strftime(self.DATE_FORMAT)
        self.LOG_DIR = 'runs'
        self.SAVE_EPOCH = 50
        self.data_name = './datasets_0425/all_data/all_data.txt'

######################################################
# ---use ShuffleNet V2 as classification model -----#
######################################################
def shuff(ratio,sel_bands_num,input_shape,times):
    args = Parameters()
    lines_train, lines_test = load_data2(args.data_name, times, ratio)
    # train cnn with all bands
    eb = 128
    net = train_cnn(args,lines_train,None,input_shape,eb,times,test_=False)
    # generate new dataset and input into the trained model, then get the ACC curve
    lines_train = lines_aug(lines_train)
    acc_or = gen_acc_curve(args, net, lines_train, input_shape,Step=3)
    # calculate mean spectrum for calculating the RBn 
    spectral_mean = cal_mean_spectral(lines_train)
    # if ACCn > threshold, set the zering step=5
    try:
        check_Step(sel_bands_num, acc_or,spectral_mean)
    except :
        acc_or = gen_acc_curve(args, net, lines_train, input_shape,Step=5)

    ### extract feature bands
    selected_bands = []
    for bands_num in range(4, sel_bands_num+1,2):
        selected_band_i = Bia_method_cnn(bands_num, acc_or,spectral_mean,)
        selected_band_i = list(np.sort(selected_band_i))
        selected_bands.append(selected_band_i)
    return selected_bands





