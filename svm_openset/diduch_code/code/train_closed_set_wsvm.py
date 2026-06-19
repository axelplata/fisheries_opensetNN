import re
import subprocess
import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

path_wsvm = '/usr/src/svm_openset/libsvm-openset/' #'/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)
    
def f1_macro(full_preds, full_labels):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  #Use only known samples to train the classifier in a closed set
  known_preds = cp_preds[cp_labels != 99]
  known_labels = cp_labels[cp_labels != 99]

  f1_macro = f1_score(known_labels, known_preds, average='macro')

  return f1_macro


def wsvm( gamma, C):
    global output, error, out, error1
    
    models_path = '/usr/src/svm_openset/diduch_code/models/closed_set/'
    dataset_path = '/usr/src/svm_openset/diduch_code/datasets/fdwe/libsvm_format/'
    
    split = 'split_0'
    train_file = dataset_path + split + '/train'
    val_file = dataset_path + split + '/val' 

    svm_type = 8 #WSVM
    '''
        -s 9 for the PI-OSVM based on one-class svms
        -s 10 for the PI-SVM based on 1-vs-rest binary svms
        -s 8 for the WSVM based on 1-vs-rest binary svms
        more info in https://github.com/ljain2/libsvm-openset/blob/master/README-libsvm-openset
        -s 7 for the  1-vs-set based on "1-vs-all" binary svms (which is the generally recommended model)
    '''

    #WSVM
    model_basename = models_path + '/wsvm/one_vs_rest'

    #WSVM
    bashCommand = './svm-train -s ' + str(svm_type) +' -a '+ str(gamma) + ' -o ' + str(C) + ' ' + train_file + ' ' + model_basename + '_G'+str(gamma)+'_C'+str(C)

    process = subprocess.Popen(bashCommand.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()
    temp_lst=[]
   
    #WSVM
    test_command = './svm-predict ' + val_file + ' ' + model_basename + '_G'+str(gamma)+'_C'+str(C)+'_one_wsvm output.csv' 

    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv('output.csv')
    labels = df.iloc[:,0]
    preds = df.iloc[:,1]
    f1_score = f1_macro(preds, labels) 

    return f1_score, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
import numpy as np

from hyperopt import hp, tpe, fmin
space = [hp.quniform('p_val',0,0.150,0.001), hp.quniform('gamma',0,10,0.01), hp.quniform('C',0,10,1)]

def tune_func(args):
    global p_val,gamma, C
    p_val = int(args[0])
    gamma = args[1]
    C = int(args[2])

    print('parameters', args)
    f1, _ = wsvm(gamma, C)
    print('f1', f1)
    return -f1

best = fmin(tune_func,space, algo=tpe.suggest, max_evals=100)
print(best)
