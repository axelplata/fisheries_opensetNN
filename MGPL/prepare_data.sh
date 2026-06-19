split=4

#images_path='/data/Open_set/datasets/paper_2024/original/'
images_path='/data/Open_set/datasets/paper_2024/original_withoutRepetition/'
rm -rf data/fdwe/train/*
rm -rf data/fdwe/val/*
rm -rf data/fdwe/test/*

cp -r $images_path/train/* data/fdwe/train/
cp -r $images_path/val/* data/fdwe/val/
cp -r $images_path/test/* data/fdwe/test/
#rm data/fdwe/train/0/226.png
python copy_unknowns.py --data_dir data/fdwe/ --split $split

