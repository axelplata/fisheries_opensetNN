import os
from data.splits import get_splits
import argparse
import shutil

#Remember, you must copy the train/val/test folders after running a experiment. Because, this script will mix instances in the val and test set for the unknowns


# This function copies all the unknown instances correponding to the unknown species (unknown_set) from the training and val/test set to the subsed folder.
# For example: if subset is val, all unknown instances from the train and test set are copied into the val set/folder
def copy(unknown_set, data_dir, subset='val'):#could be val or test
    folder2copy = 'test'
    if subset == 'test':
        folder2copy = 'val'

    for specie in unknown_set:
        count = len(os.listdir(os.path.join(data_dir, subset,str(specie)))) + 1
        files_to_copy = os.listdir(os.path.join(data_dir, 'train',str(specie)))
        for f in files_to_copy:
            new_name = str(specie) + '_train_' + str(count)+'.png'
            shutil.copyfile(os.path.join(data_dir, 'train',str(specie),f),os.path.join(data_dir, subset,str(specie),new_name))
            count+=1

        files_to_copy = os.listdir(os.path.join(data_dir, folder2copy,str(specie)))
        for f in files_to_copy:
            new_name = str(specie) + '_' + folder2copy + '_' + str(count)+'.png'
            shutil.copyfile(os.path.join(data_dir, folder2copy,str(specie),f),os.path.join(data_dir, subset,str(specie),new_name))
            count+=1


def copy_unknowns(unknown_val, unknown_test, data_dir):

    copy(unknown_val, data_dir, subset='val')

    copy(unknown_test, data_dir, subset='test')

'''
for specie in unknown_val:
        val_count = len(os.listdir(os.path.join(data_dir, 'val',str(specie)))) + 1
        files_to_copy = os.listdir(os.path.join(data_dir, 'train',str(specie)))
        for f in files_to_copy:
            new_name = str(val_count)+'.png'
            shutil.copyfile(os.path.join(data_dir, 'train',str(specie),f),os.path.join(data_dir, 'val',str(specie),new_name))
            val_count+=1

        files_to_copy = os.listdir(os.path.join(data_dir, 'test',str(specie)))
        for f in files_to_copy:
            new_name = str(val_count)+'.png'
            shutil.copyfile(os.path.join(data_dir, 'test',str(specie),f),os.path.join(data_dir, 'val',str(specie),new_name))
            val_count+=1


    for specie in unknown_test:
        test_count = len(os.listdir(os.path.join(data_dir, 'test',str(specie)))) + 1
        files_to_copy = os.listdir(os.path.join(data_dir, 'train',str(specie)))
        for f in files_to_copy:
            new_name = str(test_count)+'.png'
            shutil.copyfile(os.path.join(data_dir, 'train',str(specie),f),os.path.join(data_dir, 'test',str(specie),new_name))
            test_count+=1

        files_to_copy = os.listdir(os.path.join(data_dir, 'val',str(specie)))
        for f in files_to_copy:
            new_name = str(test_count)+'.png'
            shutil.copyfile(os.path.join(data_dir, 'val',str(specie),f),os.path.join(data_dir, 'test',str(specie),new_name))
            test_count+=1

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/fdwe/", help='fdwe data dir in MGPL source code')
    parser.add_argument('--split', type=int, default=0, help='split')

    args, _ = parser.parse_known_args()

    known_classes, unknown_val, unknown_test, _, _ = get_splits('fdwe',num_split=args.split)

    #copy val instances

    #print(known_classes, unknown_val, unknown_test)
    copy_unknowns(unknown_val, unknown_test, args.data_dir)
