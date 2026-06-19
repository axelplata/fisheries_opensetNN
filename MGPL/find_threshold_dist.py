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
    return z, labels     , dist_min, kld_min    , dist_pred, kld_pred

def accuracy(preds, labels):
    corrects = np.sum(preds == labels)
    num = len(labels)
    acc = corrects / num
    return round((acc*100), 3)

def auroc_score(inner_score, open_score):
    y_true = np.array([0] * len(inner_score) + [1] * len(open_score))
    y_score = np.concatenate([inner_score, open_score])
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    opt_threshold = thresholds[maxindex]
    return auc_score, opt_threshold

def acc_byThreshold(preds, labels, distances, threshold):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        if distances[i] < threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)

def find_bestThreshold(distances, full_preds, full_labels):
    best_thres = 0
    best_acc = 0
    for j, dist in enumerate(distances):
        full_acc  = acc_byThreshold(full_preds, full_labels, distances, dist)
        if full_acc > best_acc:
            best_acc = full_acc
            best_thres = dist

    return best_thres, best_acc



def find_bestThreshold_balAcc(scores_known, scores_unknown, known_preds, known_labels, unknown_preds, unknown_labels):
    best_thres = 0
    best_acc = 0
    distances = scores_known #np.concatenate((scores_known, scores_unknown))
    best_acc_known = 0
    best_acc_unknown = 0
    for j, dist in enumerate(distances):
        acc_known = acc_byThreshold(known_preds, known_labels, scores_known, dist)
        acc_unknown = acc_byThreshold(unknown_preds, unknown_labels, scores_unknown, dist)
        avg_acc = (acc_known+acc_unknown)/2

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_thres = dist
            best_acc_known = acc_known
            best_acc_unknown = acc_unknown

    return best_thres, round(best_acc,3)



def test_openset(model, inner_loader, open_loader):
    model.eval()
    model = model.cuda()
    z_inner, inner_label, dist_inner_min, inner_score, dist_inner_pred, inner_pred = get_output(model, inner_loader)
    z_open, open_label, dist_open_min, open_score,dist_open_pred, open_pred = get_output(model, open_loader)
   

    dist_by_specie = {}
    kl_by_specie = {}
    for i,label in enumerate(inner_label):
        if label not in dist_by_specie:
            dist_by_specie[label]=[]
            kl_by_specie[label] = []

        dist_by_specie[label].append(dist_inner_min[i])
        kl_by_specie[label].append(inner_score[i])

    for key in dist_by_specie:
        print(key, 'avg known distance: {:.03f}, std: {:.03f}'.format(np.mean(dist_by_specie[key]), np.std(dist_by_specie[key])))
        print(key, 'avg known score: {:.03f}, std: {:.03f}'.format(np.mean(kl_by_specie[key]), np.std(kl_by_specie[key])))


    dist_open_by_specie = {}
    kl_open_by_specie = {}
    for i,label in enumerate(open_pred):
        if label not in dist_open_by_specie:
            dist_open_by_specie[label]=[]
            kl_open_by_specie[label] = []

        dist_open_by_specie[label].append(dist_open_min[i])
        kl_open_by_specie[label].append(open_score[i])


    print('*'*50)
    for key in sorted(dist_open_by_specie):
        print(key, 'avg OPEN known distance: {:.03f}, std: {:.03f}'.format(np.mean(dist_open_by_specie[key]), np.std(dist_open_by_specie[key])))
        print(key, 'avg OPEN known score: {:.03f}, std: {:.03f}'.format(np.mean(kl_open_by_specie[key]), np.std(kl_open_by_specie[key])))

    if True:
        return

    #accuracy considering only known samples
    print('*'*60)
    acc_known = accuracy(inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(inner_score), np.mean(open_score)))
    print('avg known distance: {:.03f}, avg unknown distance: {:.03f}'.format(np.mean(dist_inner_min), np.mean(dist_open_min)))
    #concatenate all samples info
    full_kl_minDist = np.concatenate((inner_score, open_score)) #kl distance to the closest prototype
    full_minDist = np.concatenate((dist_inner_min, dist_open_min)) #distance to the closest prototype
    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((inner_pred, open_pred))
    
    auc_score, threshold = auroc_score(dist_inner_min, dist_open_min)
    print('*'*60)
    print('AUC Score:', round(auc_score*100,3))
    print('AUC Score threshold:', threshold)
    print('Accuracy:', acc_byThreshold(full_preds, full_labels, full_minDist, threshold))
    print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, dist_inner_min, threshold))
    print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, dist_open_min, threshold))

    print('*'*60)
    thres, acc= find_bestThreshold(full_minDist, full_preds, full_labels)
    print('Overall accuracy threshold: ', thres)
    print('Accuracy:', acc_byThreshold(full_preds, full_labels, full_minDist, thres))
    print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, dist_inner_min, thres))
    print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, dist_open_min, thres))

    print('*'*60)
    thres_avg, acc_avg = find_bestThreshold_balAcc(dist_inner_min, dist_open_min, inner_pred, inner_label, open_pred, open_label)
    print('Averaged accuracy threshold:', thres_avg)
    print('Accuracy:',acc_byThreshold(full_preds, full_labels, full_minDist, thres_avg))
    print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, dist_inner_min, thres_avg))
    print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, dist_open_min, thres_avg))

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='test.pt', help='model')


    
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
    test_openset(model, inner_loader, open_loader)

