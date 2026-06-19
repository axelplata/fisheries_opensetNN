import argparse
import os
import numpy as np
import seaborn as sn
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


SPECIES = ['P. platessa', 'S. solea', 'S. rhombus', 'E. gurnardus', 'S. maximus', 'L. limanda', 'A. radiata', 'M. merlangus', 'S. canicula']

def get_output(model, data_loader):
    z = []
    dists = []
    labels = []
    kl_divs = []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.cuda()
            z_iter, dist_iter, kl_div_iter, _ = model(images)
            z.extend(z_iter.cpu().data.numpy()) # the representation created by the encoder
            dists.extend(dist_iter.cpu().data.numpy())
            kl_divs.extend(kl_div_iter.cpu().data.numpy())
            labels.extend(label.data.numpy())
    
    z = np.array(z)
    dists = np.array(dists)
    kl_divs = np.array(kl_divs)
    labels = np.array(labels)

    # dist
    dist_reshape = dists.reshape((len(dists), model.n_classes, model.n_sub_prototypes)) 
    dist_class_min = dist_reshape.min(2)  # min dist with the propotypes
    dist_min = np.min(dist_class_min, 1)  # min dist, closest prototype
    dist_pred = np.argmin(dist_class_min, 1) # index of the min distance, in this case, the class/prediction index
    
    # kl_div
    kld_reshape = kl_divs.reshape((len(dists), model.n_classes, model.n_sub_prototypes)) 
    kld_class_min = kld_reshape.min(2)  # min kld to prototypes
    kld_min = np.min(kld_class_min, 1)
    kld_pred = np.argmin(kld_class_min, 1)

    kld_min = kld_min.astype(float)
    kld_min = np.round(kld_min,5)
    kld_pred = kld_pred.astype(float)
    kld_pred = np.round(kld_pred,5)

    #      _, inner_label, _       , inner_score, _        , inner_pred
    return z, labels     , dist_min, kld_min    , dist_pred, kld_pred


def auroc_score(inner_score, open_score):
    y_true = np.array([0] * len(inner_score) + [1] * len(open_score))
    y_score = np.concatenate([inner_score, open_score])
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    opt_threshold = thresholds[maxindex]
    print('*'*50)
    print('Openset AUROC Score')
    print('best threshold', opt_threshold)
    print('avg known score: {:.03f}, avg unknown score: {:.03f}, AUROC score {:.03f}'.format(
        np.mean(inner_score), np.mean(open_score), auc_score))
    return auc_score


def accuracy(preds, labels):
    corrects = np.sum(preds == labels)
    num = len(labels)
    acc = round((corrects / num)*100,3)
    return acc, corrects


def acc_f1_byThreshold(full_kl_minDist, full_preds, full_labels, threshold):
    cp_preds = np.copy(full_preds)
    cp_preds [full_kl_minDist > threshold] = 999
    full_acc, _ = accuracy(cp_preds, full_labels)
    f1_macro = f1_score(full_labels, cp_preds, average='macro')
    f1_weighted = f1_score(full_labels, cp_preds, average='weighted')
    return full_acc, round(f1_macro*100,3), round(f1_weighted*100,3)

def balanced_acc_byThreshold(full_kl_minDist, full_preds, full_labels, threshold):
    cp_preds = np.copy(full_preds)
    cp_preds [full_kl_minDist > threshold] = 999
    return balanced_accuracy_score(full_labels, cp_preds)


def plot_conf_matrix(full_kl_minDist, full_preds, full_labels, threshold, categories, normalize=False, open_set=False, split=0):
    
    cp_preds = np.copy(full_preds)
    indexes = [0,1,2,3]
    file_name = 'conf_matrix_closed'

    if open_set:
        cp_preds [full_kl_minDist > threshold] = int(999)
        indexes.append(999)
        categories = categories + ['Unknown']
        file_name = 'split_' + str(split)

    if normalize:
        conf_matrix = confusion_matrix(full_labels, cp_preds, labels=indexes, normalize='true')#[1:,:-1] #to remove the background
        conf_matrix = np.round(conf_matrix,2)
    else:
        conf_matrix = confusion_matrix(full_labels, cp_preds, labels=indexes)#categories)
        file_name += '_num'


    df_cm = pd.DataFrame(conf_matrix, index = categories, columns = categories)
    plt.figure(figsize = (10,7))
    sn_plot = sn.heatmap(df_cm, annot=True, cmap='Greens', annot_kws={"size": 22}, fmt='g')
    cbar = sn_plot.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    fig = sn_plot.get_figure()
    #disp.plot()
    plt.ylabel('True labels', size=26)
    plt.xlabel('Predicted labels', size=26)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20, rotation=0)
    plt.savefig('%s.%s'%(file_name, 'pdf'), bbox_inches='tight')


#macro and weighted F1-score - unknown is considered as one species
def overall_f1_scores_byThreshold(full_kl_minDist, full_preds, full_labels, threshold, known_classes):
    cp_preds = np.copy(full_preds)
    cp_preds [full_kl_minDist > threshold] = 999
    categories = [SPECIES[element] for element in known_classes] + ['Unknown']
    f1_report = classification_report(full_labels, cp_preds, target_names=categories)

    return f1_report


#average based on the accuracy by species, unknown is considered as one species
def overall_averaged_acc_byThreshold(full_kl_minDist, full_preds, full_labels, threshold):
    cp_preds = np.copy(full_preds)
    cp_preds [full_kl_minDist > threshold] = 999
    matrix = confusion_matrix(full_labels, cp_preds)
    avg_acc_per_specie = np.mean(matrix.diagonal()/matrix.sum(axis=1))

    return round(avg_acc_per_specie *100,3)


def binary_acc_byThreshold(full_kl_minDist, full_preds, full_labels, threshold):
    cp_preds = np.copy(full_preds)
    cp_full_labels = np.copy(full_labels)
    
    cp_preds [full_kl_minDist > threshold] = 999
    cp_preds [full_kl_minDist <= threshold] = 1
    cp_full_labels[cp_full_labels != 999] =1
    full_acc, _ = accuracy(cp_preds, cp_full_labels)
    return full_acc


def binary_f1(full_kl_minDist, full_preds, full_labels, threshold):
    cp_preds = np.copy(full_preds)
    cp_full_labels = np.copy(full_labels)

    cp_preds [full_kl_minDist > threshold] = 999
    cp_preds [full_kl_minDist <= threshold] = 1
    
    cp_full_labels[cp_full_labels != 999] =1
    
    f1_macro = f1_score(cp_full_labels, cp_preds, average='macro')
    f1_weighted = f1_score(cp_full_labels, cp_preds, average='weighted')

    return round(100*f1_macro,3), round(100*f1_weighted,3)
                    
def binary_metrics(distances, full_preds, full_labels, threshold):
    known_preds = full_preds[full_labels != 999]
    unknown_preds = full_preds[full_labels == 999]
    
    known_distances = distances[full_labels != 999]
    unknown_distances = distances[full_labels == 999]
    
    known_labels = full_labels [full_labels != 999]
    unknown_labels = full_labels [full_labels == 999]

    acc_known  = binary_acc_byThreshold(known_distances, known_preds, known_labels, threshold)
    acc_unknown = binary_acc_byThreshold(unknown_distances, unknown_preds, unknown_labels, threshold)

    f1_macro, f1_weighted = binary_f1(distances, full_preds, full_labels, threshold)

    #avg = (acc_known + acc_unknown)/2
    return acc_known, acc_unknown, f1_macro, f1_weighted


def plot_roc_curve(y_true, y_score):
    """
    plots the roc curve based of the probabilities
    """
    plt.figure(figsize = (10,7))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.savefig('AUC_ROC.pdf', dpi=300, bbox_inches='tight')

def auroc_score_aux(inner_score, open_score):  
    y_true = np.array([0] * len(inner_score) + [1] * len(open_score))
    y_score = np.concatenate([inner_score, open_score])
    auc_score = roc_auc_score(y_true, y_score)
    plot_roc_curve(y_true, y_score)
    return round(auc_score*100,3)

def test_openset(model, inner_loader, open_loader, threshold, known_classes, split=0):
    model.eval()
    model = model.cuda()
    _, inner_label, _, inner_score, _, inner_pred = get_output(model, inner_loader)
    _, open_label, _, open_score, _, open_pred = get_output(model, open_loader)

    threshold = round(threshold,5) 
    #accuracy considering only known samples, based on the GT we use only the known species to compute the accuracy. We consider only known classes,
    # so we do not use the threshold to define known or unknonws.
    print(30*'*','Closed set scenario', 30*'*')
    acc_known, corrects = accuracy(inner_pred, inner_label)
    #matrix = confusion_matrix(inner_label, inner_pred)
    #avg_acc = np.mean(matrix.diagonal()/matrix.sum(axis=1))
   
    categories = [SPECIES[element] for element in known_classes]

    print('instances known classes:', len(inner_pred), 'correct classified:', corrects)
    print('Closed-set Accuracy:', acc_known)
    f1_macro = f1_score(inner_label, inner_pred, average='macro')
    f1_weighted = f1_score(inner_label, inner_pred, average='weighted')
    f1_report = classification_report(inner_label, inner_pred, target_names=categories)
    print('F1-macro closed-set:', round(f1_macro*100,3))
    print('F1-weighted closed-set:', round(f1_weighted*100,3))
    print(f1_report)
    print('*'*50)

    plot_conf_matrix(None, inner_pred, inner_label, None, categories)
    plot_conf_matrix(None, inner_pred, inner_label, None, categories, normalize=True)


    #concatenate all samples info
    full_kl_minDist = np.concatenate((inner_score, open_score)) #kl distance to the closest prototype to the known (inner) and unknown (open) scores
    full_labels = np.concatenate((inner_label, open_label)).astype(int)  #labels
    full_preds = np.concatenate((inner_pred, open_pred)).astype(int) #predictions


    print(30*'*','Open set scenario', 30*'*')
    print('Number of instances:', len(full_labels))
    print('Number of known instances:', np.sum(full_labels != 999))
    print('Number of unknown instances:', np.sum(full_labels == 999))

    #accuracy of the known samples in an open-set scenario, i.e., here we used only the known samples but they can also be predicted as Unknown after applying the thresholded inference
    acc, f1_macro, f1_weighted = acc_f1_byThreshold(inner_score, inner_pred, inner_label, threshold)
    print('Known samples accuracy:', acc)
    print('Known samples F1_macro:', f1_macro)
    print('known samples F1_weighted:', f1_weighted)


    #Binary averaged accuracy. Known as class 1 and Unknown as class 999, compute the average (KACC + UKACC)/2
    acc_known, acc_unknown, f1_macro, f1_weighted = binary_metrics(full_kl_minDist, full_preds, full_labels, threshold) 
    print('\nBinary metrics')
    print('Binary known accuracy', acc_known)
    print('Binary Unknown accuracy', acc_unknown)
    print('Binary Macro f1-score:', f1_macro)
    print('Binary Weighted f1-score', f1_weighted)
    print('Binary AUC_score', auroc_score(inner_score, open_score))

    #Normal overall accuracy considering all the species + Unknown class
    full_acc, f1_macro, f1_weighted = acc_f1_byThreshold(full_kl_minDist, full_preds, full_labels, threshold)
    print('\nAll 5 species (4 Known classes + Unknown)')
    print('Global Accuracy:', full_acc)
    print('Macro F1-score', f1_macro)
    print('Weighted F1-score', f1_weighted)

    #Averaged accuracy considering all the species + Unknown class, i.e., (ACC_c1 + ACC_c2 + ... + ACC_cn + UKACC)
    #print('Averaged among all species',overall_averaged_acc_byThreshold(full_kl_minDist, full_preds, full_labels, threshold))

    #Overall macro F1-score considering all the species + Unknown class
    f1_report = overall_f1_scores_byThreshold(full_kl_minDist, full_preds, full_labels, threshold, known_classes)

    print('*'*20,'PER SPECIES', 20*'*')
    print(f1_report)



    plot_conf_matrix(full_kl_minDist, full_preds, full_labels, threshold, categories, open_set=True, split=split) 
    plot_conf_matrix(full_kl_minDist, full_preds, full_labels, threshold, categories, open_set=True, normalize=True, split=split)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--threshold', type=float, default=50, help='threshold to define unknowns')
    parser.add_argument('--model_dir', type=str, default='model.pt', help='model dir')
    parser.add_argument('--joinUnknowns', action='store_true', help='unknowns from val and test are concatenated')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    setup_seed(2021)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu

    known_classes, unknown_val, unknown_test,  known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)
    unknown_species = unknown_val
    if args.joinUnknowns:
        unknown_test = np.concatenate((unknown_val,unknown_test))
    if args.subset == 'test':
        unknown_species = unknown_test


    print('Unknown Detection Result')
    print('Dataset: {}    Split: {}'.format(args.dset, args.split))
    print('Known:', known_classes, 'Unknown val:', unknown_val, 'Unknown test:', unknown_test)
    inner_set = get_dataset(known_dataset, args.subset, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, args.subset, unknown_species, 'open')
    inner_loader = DataLoader(inner_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('Instances known:',len(inner_loader.dataset) )
    print('Instances unknown:', len(open_loader.dataset))


    model = torch.load(args.model_dir)
    test_openset(model, inner_loader, open_loader, args.threshold, known_classes, args.split)

