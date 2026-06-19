rm -rf data/fdwe/train/*
rm -rf data/fdwe/val/*
rm -rf data/fdwe/test/*

cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/train/images/* data/fdwe/train/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/val/images/* data/fdwe/val/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/test/images/* data/fdwe/test/
rm data/fdwe/train/0/226.png
python copy_unknowns.py --data_dir data/fdwe/ --split 0
python train.py --dset fdwe --split 0 --h 512 --batch_size 16


rm -rf data/fdwe/train/*
rm -rf data/fdwe/val/*
rm -rf data/fdwe/test/*

cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/train/images/* data/fdwe/train/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/val/images/* data/fdwe/val/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/test/images/* data/fdwe/test/
rm data/fdwe/train/0/226.png
python copy_unknowns.py --data_dir data/fdwe/ --split 1
#python train.py --dset fdwe --split 1
python train.py --dset fdwe --split 1 --h 512 --batch_size 16

rm -rf data/fdwe/train/*
rm -rf data/fdwe/val/*
rm -rf data/fdwe/test/*

cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/train/images/* data/fdwe/train/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/val/images/* data/fdwe/val/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/test/images/* data/fdwe/test/
rm data/fdwe/train/0/226.png
python copy_unknowns.py --data_dir data/fdwe/ --split 2
#python train.py --dset fdwe --split 2
python train.py --dset fdwe --split 2 --h 512 --batch_size 16

