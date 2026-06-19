import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataset
from data.splits import get_splits
from model import MGPLnet, train, val
from utils import setup_seed, weight_init
from utils import reset_prototypes
from tqdm import tqdm

'''
IMPORTANT: FIRST, modify the data/datasets.py file using transforms.Normalize([0, 0, 0], [1, 1, 1]). Then you can run this code
Example
    python compute_mean_std.py --dset fdwe --split 0 --h 512
'''

datasets = [
    'svhn',
    'cifar10',
    'cifar_plus_10',
    'cifar_plus_50',
    'tiny_imagenet',
    'fdwe'
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--h', type=int, default=128, help='dimension of the hidden layer')
    parser.add_argument('--k', type=int, default=3, help='number of subclusters in each class')
    parser.add_argument('--c', type=int, default=3, help='image channel')
    parser.add_argument('--temp_inter', type=float, default=0.1, help='temperature factor')
    parser.add_argument('--temp_intra', type=float, default=1, help='temperature factor')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--arch', default='resnet18', help='net arch')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--lamda', type=float, default=0.005, help='balance param between gen & dis')
    parser.add_argument('--clip', default=False, action='store_true', help='clip grad')


    args, _ = parser.parse_known_args()
    setup_seed(2022)
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu
    if not os.path.exists('./save_model/'):
        os.makedirs('./save_model/')

    known_classes, unknown_val, unknown_test, known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)

    args.num_classes = len(known_classes)

    train_set = get_dataset(known_dataset, 'train', known_classes, 'reindex')
    
    len_train = len(train_set)

    #First, modify the data/datasets.py file using transforms.Normalize([0, 0, 0], [1, 1, 1]). Then you can run this code

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
   
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])
    image_size = 0

    for inputs in tqdm(train_loader):
        imgs = inputs[0]
        if image_size == 0:
            image_size = imgs.shape[2]
        total_sum += imgs.sum(axis = [0, 2, 3])
        total_sum_square += (imgs ** 2).sum(axis = [0, 2, 3])

    count = len_train * image_size * image_size

    # mean and std
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))



