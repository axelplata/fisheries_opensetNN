import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed
from sklearn.metrics import balanced_accuracy_score

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

    #      _, inner_label, _       , inner_score, _        , inner_pred
    return z, labels, dist_min, kld_min, dist_pred, kld_pred, kld_reshape

    #return z, labels     , dist_min, kld_min    , dist_pred, kld_pred

def accuracy(preds, labels):
    corrects = np.sum(preds == labels)
    num = len(labels)
    acc = corrects / num
    return round((acc*100), 3)


def acc_byRatio(preds, labels, ratios, threshold):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        ratio = ratios[i]
        if ratio > threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)


#We compute the ratio between the 2 nearest prototypes from differents species
def compute_ratios(distances_by_prototype):
    min_distances = np.min(distances_by_prototype, 2) #min distance per species, i.e., closest prototype from each class
    sorted_min_distances = np.sort(min_distances,1) #order closest distances from different species
    ratios = np.divide(sorted_min_distances[:,0],sorted_min_distances[:,1]) #ratio between the two nearest prototypes from different species
    return ratios

#find the ratio threshold
#considering all instances
def find_bestThreshold_balAcc(known_ratios, unknown_ratios, known_preds, known_labels, unknown_preds, unknown_labels):
    thres_values = np.arange(0.5, 1, 0.001) 
    best_ratio = 0
    best_acc = 0
    
    for t in thres_values:
        acc_known = acc_byRatio(known_preds, known_labels, known_ratios, t)
        acc_unknown = acc_byRatio(unknown_preds, unknown_labels, unknown_ratios, t)
        avg_acc = (acc_known+acc_unknown)/2

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_ratio = t
        
    return round(best_ratio, 6)

def test_openset(model, inner_loader, open_loader, use_euclidean=False):
    model.eval()
    model = model.cuda()

    # dist_inner_pred is the prediction made based on euclidean distance
    # dist_inner_min is the prediction  made based on kl divergence
    z_inner, inner_label, dist_inner_score, kl_inner_score, dist_inner_pred, kl_inner_pred, kl_inner_reshape = get_output(model, inner_loader)
    z_open, open_label, dist_open_score, kl_open_score, dist_open_pred, kl_open_pred, kl_open_reshape = get_output(model, open_loader)

    #accuracy considering only known samples
    print('*'*60)
    acc_known = accuracy(kl_inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)


    known_ratios = compute_ratios(kl_inner_reshape)
    unknown_ratios = compute_ratios(kl_open_reshape)


    #concatenate all samples info
    full_ratios = np.concatenate((known_ratios, unknown_ratios)) #kl distance to the closest prototype
    inner_preds = kl_inner_pred
    open_preds = kl_open_pred
    


    print('*'*60)
    print('total instances:', len(inner_label) + len(open_label))
    print('# Known instances:', len(inner_label))
    print('# Unknown instances:',len(open_label))
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(kl_inner_score), np.mean(kl_open_score)))
    print('std known score: {:.03f}, std unknown score: {:.03f}'.format(np.std(kl_inner_score), np.std(kl_open_score)))

    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((inner_preds, open_preds))

    
    print('*'*60)
    thres_by_ratio = find_bestThreshold_balAcc(known_ratios, unknown_ratios, inner_preds, inner_label, open_preds, open_label)
    print('Averaged accuracy threshold (AK+AUK)/2:', thres_by_ratio)
    print('Accuracy:',acc_byRatio(full_preds, full_labels, full_ratios, thres_by_ratio))
    print('Known samples accuracy:', acc_byRatio(inner_preds, inner_label, known_ratios, thres_by_ratio))
    print('Unknown samples accuracy:', acc_byRatio(open_preds, open_label, unknown_ratios, thres_by_ratio))

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='test.pt', help='model')
    parser.add_argument('--euclidean', action='store_true', help='to use euclidean distance instead of kl divergence (default)')



    setup_seed(2021)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu

    known_classes, unknown_val, unknown_test,  known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)
    print('known_classes, unknown_val, unknown_test', known_classes, unknown_val, unknown_test)
    print('Unknown Detection Result')
    print('Dataset: {}    Split: {}'.format(args.dset, args.split))
    inner_set = get_dataset(known_dataset, args.subset, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, args.subset, unknown_val, 'open')
    inner_loader = DataLoader(inner_set, batch_size=1000, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=1000, shuffle=False, num_workers=4)
    
    model = torch.load(args.model_dir)
    test_openset(model, inner_loader, open_loader, args.euclidean)

