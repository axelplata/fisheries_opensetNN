rm -rf data/fdwe/train/*
rm -rf data/fdwe/val/*
rm -rf data/fdwe/test/*

cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/train/images/* data/fdwe/train/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/val/images/* data/fdwe/val/
cp -r /data/manuel_occlusionLevelByAnno/Open_set/datasets/FDWE/FDWE_dataset_crops/test/images/* data/fdwe/test/
