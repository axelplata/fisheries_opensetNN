import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed
from sklearn.metrics import balanced_accuracy_score, f1_score

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

def acc_byThreshold(preds, labels, distances, threshold, balanced=False):
    cp_preds = np.copy(preds)
    cp_preds [distances > threshold] = 999

    #for i,_ in enumerate(cp_preds):
    #    if distances[i] > threshold: # the threshold is obtained with the val set
    #        cp_preds[i] = 999 #999 is the index used for the unknown species

    if not balanced:
        return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)
    else:
        return balanced_accuracy_score(labels, cp_preds)


def find_bestThreshold(distances, full_preds, full_labels, balanced=False):
    best_thres = 0
    best_acc = 0
    for j, dist in enumerate(distances):
        full_acc  = acc_byThreshold(full_preds, full_labels, distances, dist, balanced)
        if full_acc > best_acc:
            best_acc = full_acc
            best_thres = dist

    return best_thres, best_acc



def acc_byThreshold_binary(preds, labels, distances, threshold, balanced=False):
    cp_preds = np.copy(preds)
    cp_labels = np.copy(labels)
    cp_preds[distances > threshold] = 999 #Unknown
    cp_preds[distances <= threshold] = 1 #known
    cp_labels[cp_labels != 999] = 1
    
    if not balanced:
        return accuracy(cp_preds, cp_labels)#, balanced_accuracy_score(cp_preds, labels)
    else:
        return balanced_accuracy_score(cp_labels, cp_preds)

def find_bestThreshold_binaryOS(distances, full_preds, full_labels, balanced=False):
    best_thres = 0
    best_acc = 0
    for j, dist in enumerate(distances):
        full_acc  = acc_byThreshold_binary(full_preds, full_labels, distances, dist, balanced)
        if full_acc > best_acc:
            best_acc = full_acc
            best_thres = dist

    return best_thres, best_acc

def binary_avg_accuracy(distances, full_preds, full_labels, threshold):
    known_preds = full_preds[full_labels != 999]
    unknown_preds = full_preds[full_labels == 999]
    known_distances = distances[full_labels != 999]
    unknown_distances = distances[full_labels == 999]
    known_labels = full_labels [full_labels != 999]
    unknown_labels = full_labels [full_labels == 999]

    acc_known  = acc_byThreshold_binary(known_preds, known_labels, known_distances, threshold)
    acc_unknown = acc_byThreshold_binary(unknown_preds, unknown_labels, unknown_distances, threshold)
    avg = (acc_known + acc_unknown)/2
    return acc_known, acc_unknown, avg
    

def find_bestThreshold_binaryOS_avg(distances, full_preds, full_labels):
    best_thres = 0
    best_acc = 0
    known_preds = full_preds[full_labels != 999] 
    unknown_preds = full_preds[full_labels == 999]
    known_distances = distances[full_labels != 999]
    unknown_distances = distances[full_labels == 999]
    
    known_labels = full_labels [full_labels != 999]
    unknown_labels = full_labels [full_labels == 999]

    for j, dist in enumerate(distances):
        acc_known  = acc_byThreshold_binary(known_preds, known_labels, known_distances, dist)
        acc_unknown = acc_byThreshold_binary(unknown_preds, unknown_labels, unknown_distances, dist)
        avg = (acc_known + acc_unknown)/2
        if avg > best_acc:
            best_acc = avg
            best_thres = dist

    return best_thres, best_acc

def find_bestThreshold_binaryOS_f1_macro(distances, full_preds, full_labels):
    best_thres = 0
    best_f1 = 0
    
    for j, dist in enumerate(distances):
        cp_preds = np.copy(full_preds)
        cp_labels = np.copy(full_labels)

        cp_preds[distances > dist] = 999 #Unknown
        cp_preds[distances <= dist] = 1 #known
        cp_labels[cp_labels != 999] = 1

        f1_macro = f1_score(cp_labels, cp_preds, average='macro')
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_thres = dist

    return best_thres, best_f1


def test_openset(model, inner_loader, open_loader):
    model.eval()
    model = model.cuda()
    z_inner, inner_label, dist_inner_min, inner_score, dist_inner_pred, inner_pred = get_output(model, inner_loader)
    z_open, open_label, dist_open_min, open_score,dist_open_pred, open_pred = get_output(model, open_loader)


    print('total instances:', len(inner_label) + len(open_label))
    #accuracy considering only known samples
    print('*'*60)
    acc_known = accuracy(inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(inner_score), np.mean(open_score)))

    #concatenate all samples info
    full_kl_minDist = np.concatenate((inner_score, open_score)) #kl distance to the closest prototype
    full_minDist = np.concatenate((dist_inner_min, dist_open_min)) #distance to the closest prototype
    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((inner_pred, open_pred))
    
    auc_score, threshold = auroc_score(inner_score, open_score)
    print('*'*60)
    print('AUC Score:', round(auc_score*100,3))
    print('AUC Score threshold:', threshold)
    #print('Accuracy:', acc_byThreshold(full_preds, full_labels, full_kl_minDist, threshold))
    #print('Binary Accuracy:', acc_byThreshold_binary(full_preds, full_labels, full_kl_minDist, threshold))
    #print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, inner_score, threshold))
    #print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, open_score, threshold))

    print('*'*60)
    thres, acc= find_bestThreshold(full_kl_minDist, full_preds, full_labels)
    print('Overall accuracy threshold: ', thres)
    #print('Accuracy:', acc_byThreshold(full_preds, full_labels, full_kl_minDist, thres))
    #print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, inner_score, thres))
    #print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, open_score, thres))


    print('*'*60)
    thres, acc= find_bestThreshold(full_kl_minDist, full_preds, full_labels, balanced=True)
    print('Overall balanced accuracy threshold: ', thres)
    #print('Accuracy:', acc_byThreshold(full_preds, full_labels, full_kl_minDist, thres, balanced=True))
    #print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, inner_score, thres,balanced=True))
    #print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, open_score, thres, balanced=False))


    #print('*'*60)
    #thres, acc= find_bestThreshold_binaryOS(full_kl_minDist, full_preds, full_labels, balanced=True)
    #print('Binary balanced accuracy threshold: ', thres)
    #print('Accuracy:', acc_byThreshold_binary(full_preds, full_labels, full_kl_minDist, thres, balanced=True))
    #print('Known samples accuracy:', acc_byThreshold(inner_pred, inner_label, inner_score, thres,balanced=False))
    #print('Unknown samples accuracy:', acc_byThreshold(open_pred, open_label, open_score, thres, balanced=False))


    #print('*'*60)
    #thres, acc= find_bestThreshold_binaryOS_avg(full_kl_minDist, full_preds, full_labels)
    #print('Binary averaged accuracy threshold: ', thres)
    #known_acc, unknown_acc, acc = binary_avg_accuracy(full_kl_minDist, full_preds, full_labels, thres)
    #print('Accuracy:', acc)
    #print('Known samples accuracy:', known_acc)
    #print('Unknown samples accuracy:', unknown_acc)


    print('*'*60)
    thres_f1, f1_macro = find_bestThreshold_binaryOS_f1_macro(full_kl_minDist, full_preds, full_labels)
    print('Binary F1_macro threshold:', thres_f1)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='test.pt', help='model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    
    setup_seed(2021)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu

    known_classes, unknown_val, unknown_test,  known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)
    print('known_classes, unknown_val, unknown_test', known_classes, unknown_val, unknown_test)
    print('Unknown Detection Result')
    print('Dataset: {}    Split: {}'.format(args.dset, args.split))
    inner_set = get_dataset(known_dataset, args.subset, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, args.subset, unknown_val, 'open')
    inner_loader = DataLoader(inner_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = torch.load(args.model_dir)
    test_openset(model, inner_loader, open_loader)

