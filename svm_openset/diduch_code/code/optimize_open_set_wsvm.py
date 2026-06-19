import re
import subprocess
import os
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import f1_score

path_wsvm = '/usr/src/svm_openset/libsvm-openset/' #'/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)
    
def f1_macro(full_preds, full_labels, probs, threshold):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  cp_preds[probs < threshold] = 99 #Updating predictions based on the threshold

  cp_preds [cp_preds != 99] = 1
  cp_labels[cp_labels != 99] = 1

  f1_macro = f1_score(cp_labels, cp_preds, average='macro')

  return f1_macro


def wsvm(p, gamma, C, split):
    print('Threshold', p, 'gamma', gamma,'C', C)

    models_path = '/usr/src/svm_openset/diduch_code/models/closed_set/'
    dataset_path = '/usr/src/svm_openset/diduch_code/datasets/fdwe/libsvm_format/'
    
    split = 'split_' + str(split)
    val_file = dataset_path + split + '/val'

    #WSVM
    model_basename = models_path + '/wsvm/one_vs_rest'
    output_name = 'output_wsvm.csv'

    #WSVM
    test_command = './svm-predict -o -P ' + str(p) + ' ' + val_file + ' ' + model_basename + '_G'+str(gamma)+'_C'+str(C)+'_one_wsvm ' + output_name

    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv(output_name)
    labels = df.iloc[:,0]
    preds = df.iloc[:,1]
    probs = df.iloc[:,2]
    f1_score = f1_macro(preds, labels, probs, p) 

    return f1_score, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
import numpy as np

thres_values = np.arange(0.1, 1., 0.001) #int(args[0])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=0, help='split to use')
    parser.add_argument('--gamma', type=float, default=0.001, help='Gamma svm')
    parser.add_argument('--C', type=int, default=1, help='C SVM')
    opt = parser.parse_args()
    return opt

opt = parse_opt()
gamma = opt.gamma
C = opt.C
split = opt.split

best_f1 = 0
best_p = 0

for t in thres_values:
    t = round(t,3)
    f1_binary, _ = wsvm(t, gamma, C, split)
    print('f1_binary', f1_binary)
    if f1_binary > best_f1:
        best_f1 = f1_binary
        best_p = t


print('F1 macro bianyr:', best_f1, 'threshold', best_p)
