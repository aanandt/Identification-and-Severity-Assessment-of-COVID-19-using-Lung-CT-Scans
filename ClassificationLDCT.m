stage = 2;
GPU_No=1;
%%% Lung Mask Generation for the Dataset %%%
if stage <= 0
	path='Dataset/LDCT/Test/';
	
	
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_LDCT.sh',{' '}, num2str(GPU_No), {' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});

end

%%% Finding the infection mask using the preprocessing pipeline %%%
if stage <= 1
	addpath('Src/');
	dest_dir='Preprocessed_Datasets/LDCT';
	CT_Scan_path='Dataset/LDCT/Test/';
	LungMask_path= 'TempDir/Matlab_Mask_files_Test_LDCT/';
	PreprocessPipeline_LDCT(CT_Scan_path,LungMask_path,dest_dir);
end 

%% Testing the preprocessed dataset with the trained weights %%%
if stage <= 2
	addpath('Testing/');
	data_path = 'Preprocessed_Datasets/LDCT';
	model_path = '../../../JW/DriveUpload/Classification/ModelWeights/AblationStudy/GMM+MO+B/EfficientNet-B5/saved-model-05-0.27.hdf5';
	feature_extractor = 'EfficientNetB5';
	GT_label_file = 'Dataset/LDCT/Results_LDCT.csv';
	python3_training_command= strcat('bash Testing/Testing_LDCT.sh',{' '}, num2str(GPU_No), {' '}, data_path, {' '},model_path, {' '},feature_extractor,...
		{' '},GT_label_file);
	disp(python3_training_command);
	system(python3_training_command{1});
end