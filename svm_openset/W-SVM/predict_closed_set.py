import re
import subprocess
import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

path_wsvm = '/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)
    

# Use in the paper 2024 to find the best threshold
# Binary f1_macro: known as 1 and unknown as 99
# F1-macro of known averaged with the f1-score of unknowns
def binary_f1(full_preds, full_labels):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  #Use only known samples
  known_preds = cp_preds[cp_labels != 99]
  known_labels = cp_labels[cp_labels != 99]

  #cp_preds [cp_preds != 99] = 1
  #cp_labels[full_labels != 99] = 1

  f1_macro = f1_score(known_labels, known_preds, average='macro')

  return f1_macro #round(100*f1_macro,3)


def wsvm(gamma, C):
    #global output, error, out, error1
    split = 'split_0'
    train_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/train'
    val_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/val'

    #WSVM
    models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/wsvm/'

    #PISVM
    #models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/pisvm/'

    #WSVM
    test_command ='./svm-predict ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_one_wsvm output_wsvm.csv'

    #PISVM
    #test_command ='./svm-predict ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_NAME output_pisvm.csv'


    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv('output_wsvm.csv')
    preds = df.iloc[:,1]
    labels = df.iloc[:,0]
    f1_binary = binary_f1(preds, labels) 

    #temp_lst = re.findall(r'[-+]?\d*\.\d+|\d+', metrics_string)
    #temp_lst = list(map(float, temp_lst))
    #acc = temp_lst[0]/100
    #return acc, temp_lst
    return f1_binary, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
from hyperopt import hp, tpe, fmin
space = [hp.quniform('p_val',0,0.150,0.001), hp.quniform('gamma',0,10,0.01), hp.quniform('C',0,10,1)]

# Best values for wsvm closed set {'C': 10.0, 'gamma': 0.01}
# Best values for wsvm open set {'C': 6.0, 'gamma': 0.01, 'p_val': 0.061}

gamma = 0.01 
C = 6
f1_binary, _ = wsvm(gamma, C)
print('f1_binary', f1_binary)


#def parse_opt():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--train_file', type=str, default='libsvm/train', help='libsvm format')
#    parser.add_argument('--val_file', type=str, default='libsvm/val', help='libsvm format')
#    parser.add_argument('--models_output_dir', type=str, default='libsvm_models/', help='dir to save models')
#    opt = parser.parse_args()
#    return opt


#opt = parse_opt()
