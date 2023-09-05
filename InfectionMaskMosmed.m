stage = 1;
%%% Lung Mask Generation for the Dataset %%%
if stage <= 0
	path='Dataset/MOSMED/Test/';
	GPU_No=0;
	addpath('LungMaskGeneration/');
	python3_training_command= strcat('bash LungMaskGeneration/GenerateLungMask_Mosmed_Segment.sh',{' '}, num2str(GPU_No), {' '},path);
	disp(python3_training_command);
	system(python3_training_command{1});
end


%%% Finding the infection mask using the preprocessing pipeline %%%
if stage <= 1
	
	addpath('Src/');
	dest_dir='SegmentedResults/Mask_Dicescore_Mosmed/';
	CT_Scan_path= 'TempDir/Matlab_Mask_files_Test_Mosmed/';
	%'/speech/tmp/anand/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mosmed/'
	%'/speech/tmp/anand/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mosmed_1/'
	%'TempDir/Matlab_Mask_files_Test_Mosmed/';
	LungMask_path= 'TempDir/Matlab_Mask_files_Test_Mosmed/';
	%'/speech/tmp/anand/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mosmed/'
	%'/speech/tmp/anand/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mosmed_1/'
	%'TempDir/Matlab_Mask_files_Test_Mosmed/';
	InfectionMask_GT_path='Dataset/MOSMED/Mask/';
	InfectionMask_Mosmed(CT_Scan_path,LungMask_path,InfectionMask_GT_path,dest_dir);
end