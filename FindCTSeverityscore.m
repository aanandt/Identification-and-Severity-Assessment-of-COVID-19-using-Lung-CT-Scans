stage = 1;
GPU_No=0;
%%% Lung Mask Generation for the Dataset %%%
if stage <= 0
	path='Dataset/CTSS/';
	
	
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_CTSS.sh',{' '}, num2str(GPU_No), {' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});

end

%%% Finding the CTSS using the preprocessing pipeline %%%
if stage <= 1
	addpath('Src/');
	
	CT_Scan_path='Dataset/CTSS/';
	LungMask_path='TempDir/Matlab_Mask_files_Test_CTSS/';
	GT_CTSS_score='CTSS_GT.csv';
	ChestCTseverityScore(CT_Scan_path,LungMask_path,GT_CTSS_score);
end 

