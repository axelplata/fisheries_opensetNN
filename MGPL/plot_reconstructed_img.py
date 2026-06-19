import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed
from PIL import Image
from matplotlib import pyplot as plt


def save_unNormalizedImage(image, output_name):#image must be a numpy array
    mean = np.array([0.485, 0.456, 0.406])  # mean of your dataset
    std = np.array([0.229, 0.224, 0.225])  # std of your dataset
    original_img = image#.cpu().data.numpy()
    x = original_img * std[:,None,None] + mean[:, None,None]
    x = x.transpose(1,2,0)
    plt.imshow(x)
    plt.axis('off')
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

def plot_input_reconstructed_images(model, data_loader, output_dir, counter=0):
    print('this is the counter', counter)
    recons = []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.cuda()
            z_iter, dist_iter, kl_div_iter,recon_iter = model(images)
            recons.extend(recon_iter.cpu().data.numpy())
            for i,img in enumerate(images):
                save_unNormalizedImage(img.cpu().data.numpy(), output_dir + '/img_'+ str(counter) + '.png')
                save_unNormalizedImage(recon_iter.cpu().data.numpy()[i], output_dir + '/img_'+ str(counter) + '_recon.png')
                counter+=1

    reconstructed_img = np.array(recons)

    return reconstructed_img


def reconstruct_images(model, inner_loader, open_loader, output_dir):
    model.eval()
    model = model.cuda()
    recon_inner  = plot_input_reconstructed_images(model, inner_loader, output_dir)
    recon_open = plot_input_reconstructed_images(model, open_loader, output_dir, counter=len(inner_loader.dataset))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--subset', type=str, default='val', help=' val or test to compute the results')
    parser.add_argument('--model_dir', type=str, default='model.pt', help='model dir')
    parser.add_argument('--joinUnknowns', action='store_true', help='unknowns from val and test are concatenated')
    parser.add_argument('--output_dir', type=str,default='images/', help='output dir')
    parser.add_argument('--batch_size', type=int,default=32, help='batch size')


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
    
    model = torch.load(args.model_dir)
    reconstruct_images(model, inner_loader, open_loader, args.output_dir)

