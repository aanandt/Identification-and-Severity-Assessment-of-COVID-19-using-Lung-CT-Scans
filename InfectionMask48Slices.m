stage = 1;

%%% Lung Mask Generation for the Dataset %%%
if stage <= 0
	path = 'Dataset/InfNet/Test/';
	GPU_No=0;
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_InfNet_48Slices.sh',{' '},num2str(GPU_No), {' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});

end
%%% Finding the infection mask using the preprocessing pipeline %%%

if stage <= 1
	addpath('Src/');
	dest_dir='SegmentedResults/Mask_Dicescore_InfiNet_Test48';
	CT_Scan_path='TempDir/Matlab_Mask_files_Test_InfiNetData_48Slices/';
	LungMask_path='TempDir/Matlab_Mask_files_Test_InfiNetData_48Slices/';
	InfectionMask_GT_path='Dataset/InfNet/Mask/';
	InfectionMask_InfNet48Slices(CT_Scan_path,LungMask_path,InfectionMask_GT_path,dest_dir);
end