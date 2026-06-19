import re
import subprocess
import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

path_wsvm = '/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)
    
def f1_macro(full_preds, full_labels):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  #cp_preds [cp_preds != 99] = 1   
  #cp_labels[cp_labels != 99] = 1

  f1_macro = f1_score(cp_labels, cp_preds, average='macro')

  return f1_macro


def wsvm(p_val, gamma, C):
    #global output, error, out, error1
    split = 'split_0'
    train_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/train'
    val_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/val' #or test

    #WSVM
    models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/wsvm/'

    #PISVM
    #models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/pisvm/'

    #WSVM
    test_command = './svm-predict -P '+str(p_val)+ ' ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_one_wsvm wsvm_output.csv'

    #PISVM
    #test_command = './svm-predict -P '+str(p_val)+ ' ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_MAME_ONE_VS_REST pisvm_output.csv'

    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv('output.csv')
    preds = df.iloc[:,1]
    labels = df.iloc[:,0]
    f1_score = f1_macro(preds, labels) 

    return f1_score, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
#from hyperopt import hp, tpe, fmin

#space = [hp.quniform('p_val',0,0.150,0.001)]


thres_values = np.arange(0.1, 1., 0.001) #int(args[0])
gamma = 0.01 #args[1]
C = 10 # int(args[2])
    
best_f1 = 0
best_p = 0

for t in thres_values:
    f1_binary, _ = wsvm(t, gamma, C)
    print('f1_binary', f1_binary)
    if f1_binary > best_f1:
        best_f1 = f1_binary
        best_p = t


print(best_f1, best_p)
