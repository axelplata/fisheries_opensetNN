import re
import subprocess
import os
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report,roc_auc_score

path_wsvm = '/usr/src/svm_openset/libsvm-openset/' #'/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)


#Use in the paper 2024 to find the best threshold
# Binary f1_macro and weighted: known as 1 and unknown as 99
def binary_f1(full_preds, full_labels):
  cp_preds = np.copy(full_preds)
  cp_labels = np.copy(full_labels)

  cp_preds [cp_preds != 99] = 1
  cp_labels[full_labels != 99] = 1

  f1_macro = f1_score(cp_labels, cp_preds, average='macro')
  f1_weighted = f1_score(cp_labels, cp_preds, average='weighted')

  return round(100*f1_macro,3), round(100*f1_weighted,3)

# AUCScore binary
# Binary AUCScore: known as 1 and unknown as 999
# In the case of the scores the ratio distance used by OSNN as threshold was used here
def binary_auc_score(full_labels, y_ratios):
  cp_labels = np.copy(full_labels)
  
  cp_labels[cp_labels != 99] = 1
  cp_labels[cp_labels == 99] = 0

  auc_score = roc_auc_score(cp_labels, y_ratios)
  return round(auc_score*100,3)

#Compute the accuracy/f1-macro/f1-weighted only for the known or unknown instances
def metrics_specific_group(Y_labels, Y_pred, unknown=False, unknown_label=99):
  if unknown:
    mask = Y_labels == unknown_label
  else:
    mask = Y_labels != unknown_label

  labels = Y_labels[mask]
  preds = Y_pred[mask]
  acc = round(accuracy_score(labels, preds)*100,3)
  f1_macro = round(100*f1_score(labels, preds, average='macro'),3)
  f1_weighted = round(100*f1_score(labels, preds, average='weighted'),3)
  return acc, f1_macro, f1_weighted

def open_set_metrics(y_labels, y_preds, y_ratios, threshold):
  SPECIES = ['P. platessa', 'S. solea', 'S. rhombus', 'E. gurnardus', 'S. maximus', 'L. limanda', 'A. radiata', 'M. merlangus', 'S. canicula']
  known_classes= [0,1,3,5]

  cp_preds = np.copy(y_preds)

  cp_preds[y_ratios < threshold] = 99

  # known accuracy considering only the known instances (based on the GT),
  # but at this point the predictions (after threshold) also include the unknown label
  kc_acc, f1_macro, f1_weighted = metrics_specific_group(y_labels, cp_preds, unknown=False)
  print('Known samples accuracy:', kc_acc)
  print('Known samples Macro F1:', f1_macro)
  print('Known samples Weighted F1:', f1_weighted)

  # unknown accuracy considering only the unknown instances (based on the GT),
  # but at this point the predictions (after threshold) could be classified as one the knowns species or as unknown
  ukc_acc, f1_macro, f1_weighted = metrics_specific_group(y_labels, cp_preds, unknown=True)
  print('Unknown samples accuracy:', ukc_acc)


  # Here we compute the AVERAGED binary accuracy based on a binary problem instead
  #acc_known, acc_unknown, binary_avg = binary_avg_accuracy(y_preds, y_labels)
  #print ('Binary acc_known', acc_known)
  #print ('Binary acc_Unknown', acc_unknown)
  f1_macro, f1_weighted = binary_f1(cp_preds, y_labels)
  print('OS Binary F1-macro', f1_macro)
  print('OS Binary F1-weighted', f1_weighted)

  #Binary AUC_score
  auc_score = binary_auc_score(y_labels, y_ratios)
  print('OS Binary AUC_score', auc_score)


  # Global overall accuracy considering all the species + Unknown class
  global_acc= round(accuracy_score(y_labels, cp_preds)*100,3)
  print('Global Accuracy:', global_acc)

  # Global f1-macro and f1-weighted considering all the species + Unknown class
  f1_macro = round(f1_score(y_labels, cp_preds, average='macro')*100,3)
  f1_weighted = round(f1_score(y_labels, cp_preds, average='weighted')*100,3)
  print('F1_macro (KC + UKC):', f1_macro)
  print('F1_weighted (KC + UKC):', f1_weighted)

  categories = [SPECIES[element] for element in np.unique(known_classes)] + ['Unknown']

  f1_report = classification_report(y_labels, cp_preds, target_names=categories)
  print('*'*20,'PER SPECIES', 20*'*')
  print(f1_report)


def pisvm(gamma, C, threshold,split=0, partition='val'):
    #global output, error, out, error1

    models_path = '/usr/src/svm_openset/diduch_code/models/closed_set/'
    dataset_path = '/usr/src/svm_openset/diduch_code/datasets/fdwe/libsvm_format/'

    split = 'split_' + str(split)
    val_file = dataset_path + split + '/' + partition

    #WSVM
    model_basename = models_path + '/pisvm/one_vs_rest'
    output_name = 'output_pisvm_open.csv'

    #WSVM
    test_command = './svm-predict -o -P ' + str(threshold) + ' ' + val_file + ' ' + model_basename + '_G'+str(gamma)+'_C'+str(C)+' ' + output_name

    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv(output_name)
    labels = df.iloc[:,0]
    preds = df.iloc[:,1]
    max_prob = df.iloc[:,2]
    open_set_metrics(labels, preds, max_prob, threshold)

"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=0, help='split to use')
    parser.add_argument('--gamma', type=float, default=0.001, help='Gamma svm')
    parser.add_argument('--C', type=int, default=1, help='C SVM')
    parser.add_argument('--thres', type=float, default=0.1, help='to reject unknwons')
    parser.add_argument('--partition', type=str, default='test', help='val or test')
    opt = parser.parse_args()
    return opt

opt = parse_opt()
gamma = opt.gamma
C = opt.C
split = opt.split
threshold = opt.thres
partition = opt.partition
print('Parameters: gamma', gamma, 'C', C)
print(':'*20 + partition + ':'*20)
pisvm(gamma, C, threshold, split, partition)


