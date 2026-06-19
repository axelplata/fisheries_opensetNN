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
# Averaged f1_macro
def binary_f1(full_preds, full_labels):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  #Use only known samples
  known_preds = cp_preds[cp_labels != 99]
  known_labels = cp_labels[cp_labels != 99]

  f1_macro_known = f1_score(known_labels, known_preds, average='macro')

  
  #Use only unknown samples
  unknown_preds = cp_preds[cp_labels == 99]
  unknown_labels = cp_labels[cp_labels == 99]

  unknown_preds [unknown_preds != 99] = 1
  #unknown_labels[unknown_labels != 99] = 1

  f1_macro_unknown = f1_score(unknown_labels, unknown_preds, average='macro')


  avg = (f1_macro_known+f1_macro_unknown)/2

  return avg #f1_macro #round(100*f1_macro,3)


def wsvm(p_val, gamma, C):
    global output, error, out, error1
    split = 'split_0'
    train_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/train'
    val_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/val' #or test

    #WSVM
    models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/open_set/wsvm/'

    bashCommand = './svm-train -s 8 -a '+ str(gamma) + ' -o '+str(C)+' ' + train_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)
    process = subprocess.Popen(bashCommand.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()
    temp_lst=[]

    #WSVM
    test_command = './svm-predict -o -P '+str(p_val)+ ' ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_one_wsvm wsvm_output.csv'
    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv('wsvm_output.csv')
    preds = df.iloc[:,1]
    labels = df.iloc[:,0]
    f1_binary = binary_f1(preds, labels) 

    return f1_binary, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
from hyperopt import hp, tpe, fmin
space = [hp.quniform('p_val',0,0.150,0.001), hp.quniform('gamma',0,10,0.01), hp.quniform('C',0,10,1)]

def tune_func(args):
    global p_val,gamma, C
    p_val = int(args[0])
    gamma = args[1]
    C = int(args[2])
    
    print('parameters', args)
    f1_binary, _ = wsvm(p_val, gamma, C)
    print('f1_binary', f1_binary)
    return -f1_binary


#def parse_opt():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--train_file', type=str, default='libsvm/train', help='libsvm format')
#    parser.add_argument('--val_file', type=str, default='libsvm/val', help='libsvm format')
#    parser.add_argument('--models_output_dir', type=str, default='libsvm_models/', help='dir to save models')
#    opt = parser.parse_args()
#    return opt


#opt = parse_opt()

best = fmin(tune_func,space, algo=tpe.suggest, max_evals=100)
print(best)
