MGPL
Last docker from YOLO (ultralytics/ultralytics:latest) was used

NOT SURE if the folder "MGPL" contains the last version of code. This foler was copied from another directory (the "code" folder). TO BE CHECKED

This crop is generating some problems, so I remove it before experiments --> rm data/fdwe/train/0/226.png

General:
To execute MPGL (original code in https://github.com/LiuJMzzZ/MGPL):
	1. I made some changes, so use the MGPL version inside "code" folder
		The dataset (e.g., fdwe) should be placed in "data". inside this folder the struture must be
			fdwe
				test
					0  --> this is the folder corresponding to class 0, inside the images of each fish
					1
					.
					.
					n
				train
					0
					1
					.
					.
					n
				val	0
					1
					.
					.
					n
					
		The data/dataset.py was modified to read the FDWEDataset
		In data/splits.py you can define the splits that will be used as known and unknown
		Example with 3 splits:
			fdwe = {
			    'known_set': 'fdwe',
			    'unknown_set': 'fdwe',
			    'splits': [
				{'known_classes': [0,1,3,5], 'unknown_val': [2,4], 'unknown_test':[6,7,8]},
				{'known_classes': [0,1,3,5], 'unknown_val': [6,8], 'unknown_test':[2,4,7]},
				{'known_classes': [0,1,3,5], 'unknown_val': [2,7], 'unknown_test':[4,6,8]},
			    ]
			}

		train.py was modified:
			line 76: --> n_sub_prototypes=args.k
		
		
		Do not forget to check if the input and latent space dimensions are the correct ones. Check in data/datasets.py
			- imageSize 224
For Openset:
	1. Check the prepare_data.sh file. This file will copy the images in the corresponding folder to training our models based on the splits defined before.
	
	2. We need to organize the data/images/folders first. Because, we will use all instaces from train,val and test as unknown for the Unknown classes.
	
		./prepare_data.sh
	 
	2. To train: 
		python train.py --dset fdwe --split 0
		models are store in "save_model" folder
		 
		Example: python train.py --dset fdwe --split 0 --h 512 --batch_size 4
		
		default values: (dset='fdwe', lr=0.001, batch_size=4, epoch=100, num_classes=4, h=512, k=3, c=3, temp_inter=0.1, temp_intra=1, gpu=0, arch='resnet18', split=0, lamda=0.005, clip=False)

	
		
		
	3. To define the threshold: Several thresholds are evaluated:
		- For the 2024 paper, we used the original one used in the MGPL papaer (AUC_score)
		- Averaged accuracy between known and unkown is used as threshold. (AC_KS + AC_UKS)/2
		- Binary problem, i.e, all detected as unknown as part of class 999, othersiwe class 1.
		- We can evaluate the use of F1-macro given the imbalance. However, we can discuss about the application. Is more important to detect known or unknown?
	
		
		#This scripts shows thresholds for AUC, overall accuracy, and averaged accuracy.
		python find_threshold_kl.py --dset fdwe --split 1 --subset val --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_1.pt

		#Find the threshold per specie
		python find_threshold_bySpecie_kl.py --dset fdwe --split 1 --subset val --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_1.pt
	
		#Find threshold based on the distance (kl) between prototypes and unknowns. 
		#Link unknowns with specific prototype or class.  
		#However; there are few instances (did not work well), so
		##we used all unknowns instances to define the threshold if there were not unknown instances predicted as the current species or unknown instances linked to the current prototype
		python find_threshold_byPrototype_UnknownsbyPrediction.py --dset fdwe --split 0 --subset val --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_0.pt
		
		#Find threshold based on the distance (kl) between prototypes against ALL unknowns
		python find_threshold_byPrototype_allUnknowns.py --dset fdwe --split 0 --subset val --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_0.pt
		
		#Ratio between the closest prototypes from different classes (based on OSNN)
		python find_ratio_threshold.py --dset fdwe --split 0 --subset val --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_0.pt
	
	
	4. To test:
	
	python test_ours.py --dset fdwe --split 2 --subset test --threshold 54.567482  --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_2.pt 
		--joinUnknowns
		
	python test_bySpeciesThreshold_kl.py --dset fdwe --split 2 --subset test --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_2.pt
		
	for both versions byPrototype
	python test_byPrototypeThreshold_dist.py --dset fdwe --split 2 --subset test --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_2.pt
		
	#ratio based
	python test_byRatio.py --dset fdwe --split 2 --subset test --model_dir /data/manuel_occlusionLevelByAnno/Open_set/experiments/MGPL/models_fdwe/exp1/fdwe_split_2.pt --threshold 0.964
		
	
		
	5. To plot input image and recosntructed image. If necessary, change the input size in data/dataset.py
		python plot_reconstructed_img.py --dset fdwe --split 2 --subset test --model_dir MGPL/models_fdwe/exp1_2/fdwe_split_0.pt --output_dir MGPL/models_fdwe/exp1_2/imagesTest_split2_224In_512Latent/
		
	5. To plot sne features. The tsne.py file is in the scripts folder
	 - python extract_features.py --dset fdwe --split 0 --subset test --model_dir MGPL/models_fdwe/exp1_2/fdwe_split_0.pt --features_output_file features_original_MGPL/split2_withTest2.npy
	 - python3 tsne.py --features_file ../experiments/MGPL/models_fdwe/exp1/features_original_MGPL/split2_test.npy --img_output MGPL_split2_test.pdf





LAST EXPERIMENTS
to change the input size, you should go to:

	vi data/dataset.py
	
	and update the transform_train and transform_test
	
	then the h value will update the size of the feature representation created by the network that is used for prediction
	
	
	python train.py --dset fdwe --split 2 --h 512 --batch_size 16




		

