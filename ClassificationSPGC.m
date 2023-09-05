stage = 2;
GPU_No=0;
%%% Lung Mask Generation for the Dataset %%%
if stage <= 0
	path='Dataset/SPGC/Test/';
	
	
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_SPGC_Test.sh',{' '}, num2str(GPU_No), {' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});

end

%%% Finding the infection mask using the preprocessing pipeline %%%
if stage <= 1
	addpath('Src/');
	dest_dir='Preprocessed_Datasets/SPGC';
	CT_Scan_path='Dataset/SPGC/Test/';
	LungMask_path='TempDir/Matlab_Mask_files_Test_SPGC/';
	PreprocessPipeline_SPGC(CT_Scan_path,LungMask_path,dest_dir);
end 

%% Testing the preprocessed dataset with the trained weights %%%
if stage <= 2
	addpath('Testing/');
	data_path = 'Preprocessed_Datasets/SPGC/Test/';
	model_path = '/speech/tmp/anand/Covid_2021/JW/DriveUpload/Classification/ModelWeights/AblationStudy/GMM+MO+B/EfficientNet-B5/saved-model-05-0.27.hdf5'
	
	feature_extractor = 'EfficientNetB5';
	GT_label_file = 'Dataset/SPGC/Test/final_label.csv';
	python3_training_command= strcat('bash Testing/Testing_SPGC.sh',{' '}, num2str(GPU_No), {' '}, data_path, {' '},model_path, {' '},feature_extractor,...
		{' '},GT_label_file);
	disp(python3_training_command);
	system(python3_training_command{1});
end