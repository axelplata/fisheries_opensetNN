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


#Considering the distance per prototype to all UKNOWNS to define the threshold


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

    # dist euclidean
    dist_reshape = dists.reshape((len(dists), model.n_classes, model.n_sub_prototypes))
    dist_class_min = dist_reshape.min(2)  # min dist with the propotypes
    dist_min = np.min(dist_class_min, 1)  # min dist, closest prototype
    dist_pred = np.argmin(dist_class_min, 1) # index of the min distance, in this case, the class/prediction index
    
    # kl_div
    kld_reshape = kl_divs.reshape((len(dists), model.n_classes, model.n_sub_prototypes)) 
    kld_class_min = kld_reshape.min(2)  # min kld to prototypes
    kld_min = np.min(kld_class_min, 1)
    kld_pred = np.argmin(kld_class_min, 1)
    min_byClass = kld_reshape[0].min(1)
    min_all = np.min(min_byClass)
    pred = np.argmin(min_byClass)

    return z, labels, dist_min, kld_min, dist_pred, kld_pred, kld_reshape

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


def acc_byThreshold_byPrototype(preds, labels, dist_prototypes, thresholds_by_prototype):
    cp_preds = np.copy(preds)
    for i,_ in enumerate(cp_preds):
        distance = np.min(dist_prototypes[i,cp_preds[i]])
        closest_prototype = np.argmin(dist_prototypes[i,cp_preds[i]])
        threshold = thresholds_by_prototype[cp_preds[i],closest_prototype]
        if distance > threshold: # the threshold is obtained with the val set
            cp_preds[i] = 999 #999 is the index used for the unknown species

    return accuracy(cp_preds, labels)#, balanced_accuracy_score(cp_preds, labels)



#Considering the distance per prototype to all UKNOWNS to define the threshold
def find_bestThreshold_balAcc(known_preds, known_labels, unknown_preds, unknown_labels, dist_prototypes_known, dist_prototypes_unknown):
    n_species = dist_prototypes_known.shape[1]
    n_prototypes = dist_prototypes_known.shape[2]
    threshold_by_prototype = np.zeros((n_species, n_prototypes))

    for specie in np.unique(known_labels):
        #Considering only the correct classified instances to define the threshold
        mask_labels = known_labels == specie #find instances of the current specie
        mask_correct_preds = known_preds[mask_labels] == specie #considering only correct predictions
          
        #Accuracies by prototype --> define the best boundary per prototype
        prototype_dist = dist_prototypes_known[mask_labels]
        distances_correct_preds = prototype_dist[mask_correct_preds] #all distances correct instances. This includes the distance from this instance to all prototypes from all classes
        
        distances_correct_preds = distances_correct_preds [:, specie] #only distances from this instance to all prototypes of a specific class (specie)

        min_distances = np.min(distances_correct_preds, 1)
        prototypes_used = np.argmin(distances_correct_preds, 1) #indexes of the used prototypes, i.e., the closest prototypes (min distance)

        unique_prototypes_used = np.unique(prototypes_used)

        for prototype in unique_prototypes_used:
            best_thres = 0
            best_acc = 0

            distances_prototype = min_distances[prototypes_used==prototype]
            std_dist = np.std(distances_prototype)

            known_preds_specie = known_labels_specie = np.full(len(distances_prototype), specie)

            #distance to all the unknown samples from the current prototype
            #Should I use only the unknown that were defined based on the distance with the current prototype?
            #cp_dist_prototypes_unknown = dist_prototypes_unknown[unknown_preds == specie]
            scores_unknown = dist_prototypes_unknown[:,specie,prototype] 
            for j, dist in enumerate(distances_prototype):
                acc_known = acc_byThreshold(known_preds_specie, known_labels_specie, distances_prototype, dist)
                acc_unknown = acc_byThreshold(unknown_preds, unknown_labels, scores_unknown, dist)
                avg_acc = (acc_known+acc_unknown)/2

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_thres = dist

            threshold_by_prototype[specie,prototype] = round(float(best_thres),6)
        
        threshold_zero = threshold_by_prototype[specie,:] == 0
        threshold_nozero = threshold_by_prototype[specie,:] > 0
        threshold_by_prototype[specie,:][threshold_zero] = np.min(threshold_by_prototype[specie,:][threshold_nozero])

    return threshold_by_prototype

def print_prototypes(prototypes):
    str_prototypes = [str(i) for i in prototypes]
    present = '['
    for row in str_prototypes:
        present += row.strip().replace(' ',',').replace('][','],[').replace(',]',']')
    present += ']'
    print(present.replace('][','],['))

def test_openset(model, inner_loader, open_loader, use_euclidean=False):
    model.eval()
    model = model.cuda()

    # dist refers to the euclidean distance
    # kl refers to kl divergence
    z_inner, inner_label, dist_inner_score, kl_inner_score, dist_inner_pred, kl_inner_pred, kl_reshape_inner = get_output(model, inner_loader)
    z_open, open_label, dist_open_score, kl_open_score, dist_open_pred, kl_open_pred, kl_reshape_open = get_output(model, open_loader)

    #accuracy considering only known samples
    print('*'*60)
    acc_known = accuracy(kl_inner_pred, inner_label)
    print('Accuracy closed set:', acc_known)

    #concatenate all samples info
    if use_euclidean:
        print('NOT IMPLEMENTED YET BECAUSE OF BAD RESULTS:::::::::')
        full_distances = np.concatenate((dist_inner_score, dist_open_score)) #euclidean distance to the closest prototype
        open_distances = dist_open_score
        inner_distances = dist_inner_score
        inner_preds = dist_inner_pred
        open_preds = dist_open_pred

    else:
        full_distances = np.concatenate((kl_reshape_inner, kl_reshape_open)) #kl distance to all prototypes (classes * #prototypes)
        open_distances = kl_reshape_open
        inner_distances = kl_reshape_inner
        inner_preds = kl_inner_pred
        open_preds = kl_open_pred


    print('*'*60)
    print('total instances:', len(inner_label) + len(open_label))
    print('# Known instances:', len(inner_label))
    print('# Unknown instances:',len(open_label))
    print('avg known score: {:.03f}, avg unknown score: {:.03f}'.format(np.mean(inner_distances), np.mean(open_distances)))
    print('std known score: {:.03f}, std unknown score: {:.03f}'.format(np.std(inner_distances), np.std(open_distances)))

    full_labels = np.concatenate((inner_label, open_label))
    full_preds = np.concatenate((inner_preds, open_preds))
    
    print('*'*60)
    thres_by_prototype = find_bestThreshold_balAcc(inner_preds, inner_label, open_preds, open_label, inner_distances, open_distances)
    print('Averaged accuracy used, threshold by prototype and specie:', print_prototypes(thres_by_prototype))
    print('Accuracy:', acc_byThreshold_byPrototype(full_preds, full_labels, full_distances, thres_by_prototype))
    print('Known samples accuracy:', acc_byThreshold_byPrototype(inner_preds, inner_label, inner_distances, thres_by_prototype))
    print('Unknown samples accuracy:', acc_byThreshold_byPrototype(open_preds, open_label, open_distances, thres_by_prototype))

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

