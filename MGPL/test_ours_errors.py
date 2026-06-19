import argparse
import os
import numpy as np
import seaborn as sn
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from torch.utils.data import DataLoader
from data.dataset_cp import get_dataset
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
    img_paths = []
    with torch.no_grad():
        for images, label, path in data_loader:
            images = images.cuda()
            z_iter, dist_iter, kl_div_iter, _ = model(images)
            z.extend(z_iter.cpu().data.numpy()) # the representation created by the encoder
            dists.extend(dist_iter.cpu().data.numpy())
            kl_divs.extend(kl_div_iter.cpu().data.numpy())
            labels.extend(label.data.numpy())
            img_paths.extend(path)
    

    z = np.array(z)
    dists = np.array(dists)
    kl_divs = np.array(kl_divs)
    labels = np.array(labels)
    img_paths = np.array(img_paths)
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
    return z, labels     , dist_min, kld_min    , dist_pred, kld_pred, img_paths

def check_false_positives_image_path(labels, preds, categories, img_paths):
    indexes = np.unique(labels)
    indexes [indexes == 999] = len(categories)-1
    preds [preds == 999] = len(categories)-1
    
    for i in indexes:
        print('Predictions made for',i, categories[i])
        

        preds_i = preds[labels == i]
        img_path_i = img_paths [labels == i]
        errors = preds_i [preds_i != i]
        img_path = img_path_i [preds_i != i]

        print('::::Errors::::::::')
        for s in np.unique(errors):
            paths = img_path[errors == int(s)]
            print(len(paths),'predicted as', categories[int(s)])
            print(paths)

        print('*'*30)

def test_openset(model, inner_loader, open_loader, threshold, known_classes):
    model.eval()
    model = model.cuda()
    _, inner_label, _, inner_score, _, inner_pred, inner_paths = get_output(model, inner_loader)
    _, open_label, _, open_score, _, open_pred, open_paths = get_output(model, open_loader)

    threshold = round(threshold,5) 
    #accuracy considering only known samples, based on the GT we use only the known species to compute the accuracy. We consider only known classes,
    # so we do not use the threshold to define known or unknonws.
    print(30*'*','Closed set scenario', 30*'*')
    #matrix = confusion_matrix(inner_label, inner_pred)
    #avg_acc = np.mean(matrix.diagonal()/matrix.sum(axis=1))
   
    categories = [SPECIES[element] for element in known_classes]

    check_false_positives_image_path(inner_label, inner_pred, categories, inner_paths)



    #concatenate all samples info
    full_kl_minDist = np.concatenate((inner_score, open_score)) #kl distance to the closest prototype to the known (inner) and unknown (open) scores
    full_labels = np.concatenate((inner_label, open_label)).astype(int)  #labels
    full_preds = np.concatenate((inner_pred, open_pred)).astype(int) #predictions
    full_paths = np.concatenate((inner_paths, open_paths))

    print(30*'*','Open set scenario', 30*'*')
    categories.append('Unknown') 
    full_preds [full_kl_minDist > threshold] = 999
    check_false_positives_image_path(full_labels, full_preds, categories, full_paths)

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
    test_openset(model, inner_loader, open_loader, args.threshold, known_classes)

