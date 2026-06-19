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

    #      _, inner_label, _       , inner_score, _        , inner_pred
    return z, labels     , dist_min, kld_min    , dist_pred, kld_pred

def accuracy(preds, labels):
    corrects = np.sum(preds == labels)
    num = len(labels)
    acc = corrects / num
    return round((acc*100), 3)

def acc_byThreshold(preds, labels, distances, threshold):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        if distances[i] > threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)


def acc_byThreshold_bySpecie(preds, labels, distances, thresholds_by_specie):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        threshold = thresholds_by_specie[cp_preds[i]]
        #print('thr being used', threshold, 'for:', cp_preds[i])
        if distances[i] > threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)



#find best threshold by specie
#considering only corrected classified
def find_bestThreshold_avgAcc(scores_known, scores_unknown, known_preds, known_labels, unknown_preds, unknown_labels):
    threshold_by_specie = {}
    # Accuracies computed by specie, try to get better boundaries per specie 
    for specie in np.unique(known_labels):
        best_thres = 0
        best_acc = 0

        #Considering only the correct classified instances to define the threshold
        mask_labels = known_labels == specie #find instances of the current specie
        mask_correct_preds = known_preds[mask_labels] == specie #which predictions are correct
        dist_specie_preds = scores_known[mask_labels] #distances
        distances = dist_specie_preds[mask_correct_preds] #distances of the corrected classified distances
        known_preds_specie = known_labels_specie = np.full(len(distances), specie) # all preds

        for j, dist in enumerate(distances):
            acc_known = acc_byThreshold(known_preds_specie, known_labels_specie,distances, dist)
            acc_unknown = acc_byThreshold(unknown_preds, unknown_labels, scores_unknown, dist)
            avg_acc = (acc_known+acc_unknown)/2

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_thres = dist
        
        threshold_by_specie[specie] = round(float(best_thres),6)
    return threshold_by_specie



#considering only corrected classified
def find_bestThreshold_binary_f1_macro(scores_known, scores_unknown, known_preds, known_labels, unknown_preds, unknown_labels):
    threshold_by_specie = {}
    # Accuracies computed by specie, try to get better boundaries per specie 
    for specie in np.unique(known_labels):
        best_thres = 0
        best_f1 = 0

        #Considering only the correct classified instances to define the threshold
        mask_labels = known_labels == specie #find instances of the current specie
        mask_correct_preds = known_preds[mask_labels] == specie #which predictions are correct
        dist_specie_preds = scores_known[mask_labels] #distances
        distances = dist_specie_preds[mask_correct_preds] #distances of the corrected classified distances
        known_preds_specie = known_labels_specie = np.full(len(distances), specie) # all preds

        for j, dist in enumerate(distances):
            cp_preds = np.copy(known_preds_specie)
            cp_labels = np.copy(known_labels_specie)
            cp_unknown_preds = np.copy(unknown_preds)

            cp_preds[distances > dist] = 999 #Unknown
            cp_preds[distances <= dist] = 1 #known
            cp_labels[cp_labels != 999] = 1
            cp_unknown_preds[cp_unknown_preds != 999] =1

            full_labels = np.concatenate((cp_labels, unknown_labels))
            full_preds = np.concatenate((cp_preds,unknown_preds))
            f1_macro = f1_score(full_labels, full_preds, average='macro')

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_thres = dist

        threshold_by_specie[specie] = round(float(best_thres),6)
    return threshold_by_specie


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


def test_openset(model, inner_loader, open_loader, use_euclidean=False):
    model.eval()
    model = model.cuda()

    # dist_inner_pred is the prediction made based on euclidean distance
    # dist_inner_min is the prediction  made based on kl divergence
    _, inner_label, _, kl_inner_score, _, kl_inner_pred = get_output(model, inner_loader)
    _, open_label, _, kl_open_score, _, kl_open_pred = get_output(model, open_loader)

    #accuracy considering only known samples
    print('*'*60)
    acc_known = accuracy(kl_inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)

    #concatenate all samples info
    full_distances = np.concatenate((kl_inner_score, kl_open_score)) #kl distance to the closest prototype

    print('*'*60)
    print('total instances:', len(inner_label) + len(open_label))
    print('# Known instances:', len(inner_label))
    print('# Unknown instances:',len(open_label))
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(kl_inner_score), np.mean(kl_open_score)))
    print('std known score: {:.03f}, std unknown score: {:.03f}'.format(np.std(kl_inner_score), np.std(kl_open_score)))

    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((kl_inner_pred, kl_open_pred))
    
    print('*'*60)
    thres_by_specie = find_bestThreshold_avgAcc(kl_inner_score, kl_open_score, kl_inner_pred, inner_label, kl_open_pred, open_label)
    print('Averaged accuracy threshold (AK+AUK)/2:', thres_by_specie)
    print('Accuracy:', acc_byThreshold_bySpecie(full_preds, full_labels, full_distances, thres_by_specie))
    print('Known samples accuracy:', acc_byThreshold_bySpecie(kl_inner_pred, inner_label, kl_inner_score, thres_by_specie))
    print('Unknown samples accuracy:', acc_byThreshold_bySpecie(kl_open_pred, open_label, kl_open_score, thres_by_specie))


    thres_by_specie = find_bestThreshold_binary_f1_macro(kl_inner_score, kl_open_score, kl_inner_pred, inner_label, kl_open_pred, open_label)
    print('F1_macro threshold:', thres_by_specie)
    print('Accuracy:', acc_byThreshold_bySpecie(full_preds, full_labels, full_distances, thres_by_specie))
    print('Known samples accuracy:', acc_byThreshold_bySpecie(kl_inner_pred, inner_label, kl_inner_score, thres_by_specie))
    print('Unknown samples accuracy:', acc_byThreshold_bySpecie(kl_open_pred, open_label, kl_open_score, thres_by_specie))

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='test.pt', help='model')
    parser.add_argument('--euclidean', action='store_true', help='to use euclidean distance instead of kl divergence (default)')
    parser.add_argument('--batch_size', type=int, default=16,  help='batch size')


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
    test_openset(model, inner_loader, open_loader, args.euclidean)

