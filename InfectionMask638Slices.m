stage = 1;

%%% Lung Mask Generation for the Dataset %%%
if stage <= 0

	path = 'Dataset/MEDSEG/Test/';
	GPU_No=0;
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_MedSeg.sh',{' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});
end

%%% Finding the infection mask using the preprocessing pipeline %%%

if stage <= 1
	addpath('Src/');
	dest_dir='SegmentedResults/Mask_Dicescore_InfiNet_Test638';
	CT_Scan_path='TempDir/Matlab_Mask_files_Test_Medseg/';
	LungMask_path='TempDir/Matlab_Mask_files_Test_Medseg/';
	InfectionMask_GT_path='Dataset/MEDSEG/Mask/';
	InfectionMask_InfNet638Slices(CT_Scan_path,LungMask_path,InfectionMask_GT_path,dest_dir);
end