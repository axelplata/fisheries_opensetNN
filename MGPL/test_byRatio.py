import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed


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
    return z, labels     , dist_min, kld_min    , dist_pred, kld_pred, kld_reshape



def accuracy(preds, labels):
    corrects = np.sum(preds == labels)
    num = len(labels)
    acc = corrects / num
    return round(acc*100,3)


def acc_byRatio(preds, labels, ratios, threshold):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        ratio = ratios[i]
        if ratio > threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)

def compute_ratios(distances_by_prototype):
    min_distances = np.min(distances_by_prototype, 2)
    sorted_min_distances = np.sort(min_distances,1)
    ratios = np.divide(sorted_min_distances[:,0],sorted_min_distances[:,1])
    return ratios

def test_openset(model, inner_loader, open_loader, threshold_by_ratio):
    model.eval()
    model = model.cuda()
    _, inner_label, _, inner_score, _, inner_pred, kl_inner_reshape = get_output(model, inner_loader)
    _, open_label, _, open_score, _, open_pred, kl_open_reshape = get_output(model, open_loader)
    
    #accuracy considering only known samples
    acc_known = accuracy(inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)
    #auroc = auroc_score(inner_score, open_score)

    known_ratios = compute_ratios(kl_inner_reshape)
    unknown_ratios = compute_ratios(kl_open_reshape)

    #concatenate all samples info
    full_ratios = np.concatenate((known_ratios, unknown_ratios)) #kl distance to the closest prototype

    print('*'*60)
    print('total instances:', len(inner_label) + len(open_label))
    print('# Known instances:', len(inner_label))
    print('# Unknown instances:',len(open_label))
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(inner_score), np.mean(open_score)))
    print('std known score: {:.03f}, std unknown score: {:.03f}'.format(np.std(inner_score), np.std(open_score)))

    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((inner_pred, open_pred))


    print('*'*60)
    print('Averaged accuracy threshold:', threshold_by_ratio)
    print('Accuracy:',acc_byRatio(full_preds, full_labels, full_ratios, threshold_by_ratio))
    print('Known samples accuracy:', acc_byRatio(inner_pred, inner_label, known_ratios, threshold_by_ratio))
    print('Unknown samples accuracy:', acc_byRatio(open_pred, open_label, unknown_ratios, threshold_by_ratio))



#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='model.pt', help='model dir')
    parser.add_argument('--joinUnknowns', action='store_true', help='unknowns from val and test are concatenated')
    parser.add_argument('--threshold', type=float, help='ratio threshold')
    
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
    inner_loader = DataLoader(inner_set, batch_size=1000, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=1000, shuffle=False, num_workers=4)
    
    model = torch.load(args.model_dir)
    test_openset(model, inner_loader, open_loader, args.threshold)

