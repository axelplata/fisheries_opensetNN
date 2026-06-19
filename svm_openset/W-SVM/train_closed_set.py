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

  #Use only known samples to train the classifier in a closed set
  known_preds = cp_preds[cp_labels != 99]
  known_labels = cp_labels[cp_labels != 99]

  f1_macro = f1_score(known_labels, known_preds, average='macro')

  return f1_macro


def wsvm( gamma, C):
    global output, error, out, error1
    split = 'split_0'
    train_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/train'
    val_file = '/usr/src/svm_openset/W-SVM/datasets/fdwe/libsvm_format/' + split + '/val'
    #WSVM
    models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/wsvm/'
    #PISVM
    #models_dir = '/usr/src/svm_openset/W-SVM/models/optimize_param/'+ split + '/closed_set/pisvm/'

    #WSVM
    bashCommand = './svm-train -s 8 -a '+ str(gamma) + ' -o '+str(C)+' ' + train_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)

    #PISVM
    #bashCommand = './svm-train -s 10 -a '+ str(gamma) + ' -o '+str(C)+' ' + train_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)

    process = subprocess.Popen(bashCommand.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()
    temp_lst=[]
    
    #WSVM
    test_command = './svm-predict ' + val_file + ' ' + models_dir + 'model_G'+str(gamma)+'_C'+str(C)+'_one_wsvm output.csv'

    print('after test comand::::::::::::::::::::::::::::::::')

    #PISVM
    #test_command = './svm-predict ' + val_file + ' ' + models_dir + 'model_'+str(gamma)+'_'+str(C)+'_NAME output.csv'

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
from hyperopt import hp, tpe, fmin
#space = [hp.quniform('p_val',0,0.150,0.001), hp.quniform('gamma',0,10,0.01), hp.quniform('C',0,10,1)]
space = [hp.quniform('gamma',0,10,0.01), hp.quniform('C',0,10,1)]


def tune_func(args):
    global gamma, C
    #p_val = int(args[0])
    gamma = args[0]
    C = int(args[1])
    
    print('parameters', args)
    f1_binary, _ = wsvm(gamma, C)
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
