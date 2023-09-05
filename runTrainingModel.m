stage = 2; 
%%% Lung Mask Generation for the Dataset %%%
GPU_No = 1;
if stage <= 0

	path = 'Dataset/SPGC/Train/';
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_SPGC_Train.sh',{' '}, num2str(GPU_No),{' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});
end

%%% Finding the infection mask using the preprocessing pipeline %%%

if stage <= 1
	
	addpath('Src/');
	dest_dir='Preprocessed_Datasets/SPGC/';
	CT_Scan_path='Dataset/SPGC/';
	LungMask_path='TempDir/Matlab_Mask_files_Train_SPGC/';
	Slice_Label_file='Dataset/SPGC/Slice_level_labels_dummy.csv';
	PreprocessPipeline_SPGC_Train(CT_Scan_path,LungMask_path,Slice_Label_file, dest_dir);
end


% %%% Training the model with the preprocessed datasets %%%
if stage <= 2 
	
	addpath('TrainModel/');
	python3_training_command= strcat('bash TrainModel/Train_main.sh');
	disp(python3_training_command);
	system(python3_training_command);
end
