import re
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

path_wsvm = '/usr/src/svm_openset/libsvm-openset/' #'/usr/src/svm_openset/W-SVM/W-SVM/' #path to W-SVM directory
if os.getcwd() != path_wsvm:
    os.chdir(path_wsvm)


def plot_conf_matrix(preds, labels, known_classes, species, normalize=False, open_set=False, split=0):

  indexes = np.copy(known_classes)
  categories = []
  file_name = 'conf_matrix_closed'
  for i in known_classes:
    categories.append(species[i])

  if open_set:
    indexes = np.append(indexes, [99])
    categories = categories + ['Unknown']
    file_name = 'split_' + str(split)

  if normalize:
    conf_matrix = confusion_matrix(labels, preds, labels=indexes, normalize='true')#[1:,:-1] #to remove the background
    conf_matrix = np.round(conf_matrix,2)
  else:
    conf_matrix = confusion_matrix(labels, preds, labels=indexes)#categories)
    file_name += '_num'

  df_cm = pd.DataFrame(conf_matrix, index = categories, columns = categories)
  plt.figure(figsize = (10,7))
  sn_plot = sn.heatmap(df_cm, annot=True, cmap='Greens', annot_kws={"size": 22}, fmt='g')
  cbar = sn_plot.collections[0].colorbar
  # here set the labelsize by 20
  cbar.ax.tick_params(labelsize=20)

  fig = sn_plot.get_figure()
  plt.ylabel('True labels', size=26)
  plt.xlabel('Predicted labels', size=26)
  plt.xticks(fontsize=20, rotation=90)
  plt.yticks(fontsize=20, rotation=0)
  plt.savefig('%s.%s'%(file_name, 'pdf'), bbox_inches='tight')

#Computes accuracies/f1-macro/f1-weighted among KNOWN samples
def closed_set_metrics(y_labels, y_preds):
  
  SPECIES = ['P. platessa', 'S. solea', 'S. rhombus', 'E. gurnardus', 'S. maximus', 'L. limanda', 'A. radiata', 'M. merlangus', 'S. canicula']
  known_classes= [0,1,3,5]

  #We use only the known species (label != 99)
  closed_y = y_labels[y_labels != 99]
  closed_preds = y_preds[y_labels != 99]

  acc= round(accuracy_score(closed_y, closed_preds)*100,3)

  matrix = confusion_matrix(closed_y, closed_preds)
  avg_acc = round(100*np.mean(matrix.diagonal()/matrix.sum(axis=1)),3)

  f1_macro = round(100*f1_score(closed_y, closed_preds, average='macro'),3)
  f1_weighted = round(100*f1_score(closed_y, closed_preds, average='weighted'),3)

  print('Closed-set Accuracy:', acc)
  print('avg closed-set acc', avg_acc)
  print('F1-macro:', f1_macro)
  print('F1-weighted:', f1_weighted)


  categories = [SPECIES[element] for element in np.unique(known_classes)]

  f1_report = classification_report(closed_y, closed_preds, target_names=categories)
  print('*'*20,'PER SPECIES', 20*'*')
  print(f1_report)

  plot_conf_matrix(y_preds, y_labels, known_classes, SPECIES, normalize=False, split=0)
  plot_conf_matrix(y_preds, y_labels, known_classes, SPECIES, normalize=True, split=0)

def wsvm(gamma, C, partition='val'):
    #global output, error, out, error1

    models_path = '/usr/src/svm_openset/diduch_code/models/closed_set/'
    dataset_path = '/usr/src/svm_openset/diduch_code/datasets/fdwe/libsvm_format/'

    split = 'split_0'
    val_file = dataset_path + split + '/' + partition

    #WSVM
    model_basename = models_path + '/pisvm/one_vs_rest'
    output_name = 'output_pisvm.csv'

    #WSVM
    test_command = './svm-predict ' + val_file + ' ' + model_basename + '_G'+str(gamma)+'_C'+str(C)+' ' + output_name

    process1 = subprocess.Popen(test_command.split(),stdout = subprocess.PIPE)
    out, error1 = process1.communicate()
    metrics_string = out.decode()

    df = pd.read_csv(output_name)
    preds = df.iloc[:,1]
    labels = df.iloc[:,0]
    closed_set_metrics(labels, preds)
    #f1_binary = binary_f1(preds, labels) 

    #temp_lst = re.findall(r'[-+]?\d*\.\d+|\d+', metrics_string)
    #temp_lst = list(map(float, temp_lst))
    #acc = temp_lst[0]/100
    #return acc, temp_lst
    #return f1_binary, None # acc, temp_lst
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
from hyperopt import hp, tpe, fmin

# Best values for pisvm closed sed {'C': 4.0, 'gamma': 2.91}
# Version 2 {'C': 1.0, 'gamma': 4.87}

gamma = 2.91
C = 4
partition='test'
print('Parameters: gamma', gamma, 'C', C)
print(':'*20 + partition + ':'*20)
wsvm(gamma, C, partition)


#def parse_opt():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--train_file', type=str, default='libsvm/train', help='libsvm format')
#    parser.add_argument('--val_file', type=str, default='libsvm/val', help='libsvm format')
#    parser.add_argument('--models_output_dir', type=str, default='libsvm_models/', help='dir to save models')
#    opt = parser.parse_args()
#    return opt


#opt = parse_opt()
