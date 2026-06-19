import re
import subprocess
import os

path_wsvm = '/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)
    
def wsvm(p_val, gamma, C):
    global output, error, out, error1
    bashCommand = './svm-train -s 8 -a '+ str(gamma) + ' -o '+str(C)+' /usr/src/svm_openset/W-SVM/datasets/letters/libsvm_format/train ../models/optimize_param/model_'+str(gamma)+'_'+str(C)
    process = subprocess.Popen(bashCommand.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()
    temp_lst=[]
    test_command = './svm-predict -P '+str(p_val)+ ' /usr/src/svm_openset/W-SVM/datasets/letters/libsvm_format/test ../models/optimize_param/model_'+str(gamma)+'_'+str(C)+'_one_wsvm output.csv'
    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()
    temp_lst = re.findall(r'[-+]?\d*\.\d+|\d+', metrics_string)
    temp_lst = list(map(float, temp_lst))
    print(temp_lst)
    acc = temp_lst[0]/100
    return acc,temp_lst
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
    
    print(args)
    accuracy, _ = wsvm(p_val, gamma, C)
    return -accuracy

best = fmin(tune_func,space, algo=tpe.suggest, max_evals=100)
print(best)
