clear all;
close all;
clc;

img_filename = 'TempDir/Matlab_Mask_files_Test_Mosmed/';
mask_seg_filename = 'TempDir/Matlab_Mask_files_Test_Mosmed/';
dest_dir = 'Dump/Mosmed/';
%mask_gt_filename = '/speech/tmp/anand/Covid_Dataset/MY_TestData/COVID19_Mosmed/masks';%'/speech/tmp/anand/Covid_Dataset/MY_TestData/LungSegDataset/Infection_Mask';%

%img_filename = 'TempDir/Matlab_Mask_files_Test_LungSeg_Contour/';
%mask_seg_filename = 'TempDir/Matlab_Mask_files_Test_LungSeg_Contour/';
%mask_gt_filename = '/speech/tmp/anand/Covid_Dataset/MY_TestData/LungSegDataset/Lung_Mask';

%img_filename = 'TempDir/Matlab_Mask_files_Test_MedSeg/tr_im.nii.gz/';
%mask_seg_filename = 'TempDir/Matlab_Mask_files_Test_MedSeg/tr_im.nii.gz/';
%mask_gt_filename = '/speech/tmp/anand/Covid_Dataset/MY_TestData/MedSeg_dataset/MedSeg_dataset_1/tr_mask.nii.gz';


Mask_GT_files = dir(fullfile(mask_seg_filename));
Patientwise_volume_score = {};
pat_index = 1;
all_count =1;
mask_counter = 1;
volumeofInfection = [];
for i = 3: 1: numel(Mask_GT_files)

    patient_name = Mask_GT_files(i).name
    slice_index = 1;
    filename = [patient_name]; 
    imagepath = fullfile(mask_seg_filename, filename, 'image.mat');
    mask_seg_path = fullfile(mask_seg_filename, filename, 'mask.mat');
    Mask= load(mask_seg_path).data();
    [len, row, col] = size(Mask)
    for index = 1: 1: len
        
        
        
        curr_mask = squeeze(double(Mask(index, : , :)));

        flag = 0;
        lung_area = nnz(curr_mask);
        total_area = 512 * 512;

        lung_involvement = ((lung_area * 100)/total_area);
        if lung_involvement >= 7

            flag = 1;
        end
        start_ = 1;
        end_ = len;
        
        if (index >= start_ & index <= end_ & flag)
            

            Infection_Mask = InfectionMask_Generation(imagepath, mask_seg_path, index);
            
            
            rot_mask = imrotate(Infection_Mask, 90);
            Infection_Mask = flipdim(rot_mask, 1);
            
            
            dest_dir_1 = fullfile(dest_dir, patient_name, 'Predicted_Mask');
            if ~exist(dest_dir_1, 'dir')
                mkdir(dest_dir_1)
            end
            
                      
            dest_dir_3 = fullfile(dest_dir, patient_name, 'Original_Image');
            if ~exist(dest_dir_3, 'dir')
                mkdir(dest_dir_3)
            end
            
            filename_  = [dest_dir_1,'/Pred_mask_',num2str(index),'.png'];
            imwrite(Infection_Mask, filename_)
            
            
            filename_  = [dest_dir_3,'/image_',num2str(index),'.png'];
            niftifiles = load(imagepath).data();%niftiread(fullfile(filename));%

            volume = double(squeeze(niftifiles(index,:,:)));
            min_ = min(min(volume));
            max_ = max(max(volume));
            volume(volume < min_) = min_;
            volume(volume > max_) = max_;
            volume = (volume - min_) / (max_ - min_);
            volume = uint8(volume.*255);
            imwrite((volume), filename_);         
                
            all_count = all_count + 1;
            if (nnz(Infection_Mask) > 0)
            	volumeofInfection(slice_index) = (nnz(Infection_Mask))/(nnz(curr_mask));
            	slice_index = slice_index + 1;
            end
            
        end
    end
    Patientwise_volume_score{pat_index, 1} = (mean(volumeofInfection) * 100);
    pat_index = pat_index + 1;
end
function[Infection_mask] = My_Preprocessing(Imagepath, Maskpath, index)
    
    niftifiles = load(Imagepath).data();%niftiread(fullfile(filename));%
    Mask= load(Maskpath).data();
    Image = double(squeeze(niftifiles(index,:,:)));
    curr_mask = squeeze(double(Mask(index, : , :)));
    Image(Image <= -1024)= -1024;
    Image(Image >= 300)= 300;
    seg_Image = Image .* curr_mask;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S = reshape(seg_Image,[],1);
    max_val = max(max(seg_Image));
    min_val = min(min(seg_Image));
    num_bins = ceil((max_val - min_val)/3);
    S = S(S ~= 0);
    try
        
        GMModel = fitgmdist(S,3,'RegularizationValue',0.1);
    catch exception
        disp('There was an error fitting the Gaussian mixture model')
        error = exception.message
        
    end  
    
    img_reco = reshape(seg_Image,[],1);
    clusterX = cluster(GMModel,img_reco);
    [mean, index] = sort(GMModel.mu(:));
    std = sort(GMModel.Sigma(:));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    new_img = seg_Image;
    lower_val = (mean(2)- (1.5 * sqrt(GMModel.Sigma(index(2)))));
    upper_val = (mean(2) + (1.5 * sqrt(GMModel.Sigma(index(2)))));
    if upper_val <= 50
        new_img(new_img >= upper_val) = 0;
    else 
        new_img(new_img >= 50) = 0;
    end
    if lower_val >= -650
        new_img(new_img <= lower_val) = 0;
    else 
        new_img(new_img <= -650) = 0;
    end
    new_img(new_img ~= 0 ) = 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    seg_Image(seg_Image ==0)=-1024;
    new_img_1 = seg_Image;
    lower_val = (mean(3)- (1.5 * sqrt(GMModel.Sigma(index(3)))));
    upper_val = (mean(3) + (1.5 * sqrt(GMModel.Sigma(index(3)))));
    if upper_val <= 50
        new_img_1(new_img_1 >= upper_val) = 0;
    else 
        new_img_1(new_img_1 >= 50) = 0;
    end
    if lower_val >= -650
        new_img_1(new_img_1 <= lower_val) = 0;
    else 
        new_img_1(new_img_1 <= -650) = 0;
    end
    new_img_1(new_img_1 ~= 0 ) = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    final_mask = ceil((new_img + new_img_1) / 2);
    V1 = vesselness2D((seg_Image), 0.5:0.5:2.5, [1;1], 1, true);
    V1(V1 < 0.65) = 0;
    V1(V1 ~= 0) = 1;
    final_mask = final_mask .* ~V1;
    final_mask = bwareaopen(final_mask, 50);
    se = strel('disk', 2);
    final_mask = imfill(final_mask, 'holes');

    segmented_img = final_mask .* seg_Image;
    
    
    
    Infection_mask = segmented_img;
    Infection_mask(Infection_mask == -1024) = 0;
    Infection_mask(Infection_mask ~= 0) = 1;        
    se = strel('disk', 3);
    Infection_mask = imdilate(Infection_mask,se);
end

function [first_row, last_row] = Find_MinMaxRowIndices(masks)
    [len, row, col] = size(masks);
    first_row = 512;
    last_row = 1;
    for i = 1: 1 : len
        curr_mask =  squeeze(double(masks(i,:,:)));
        [temp_first_row, temp_last_row] = Find_RowIndices(curr_mask);
        if (temp_first_row < first_row)
            first_row = temp_first_row;
        end
        if(temp_last_row > last_row)
            last_row = temp_last_row;
        end

    end
    
end
function severity = Find_SeverityScore(curr_mask, first_row, last_row, Preprocessed_Image)
%     LLL=[];
%     LUL=[];
%     RUL=[];
%     RML=[];
%     RLL=[];

    right_lung_increment = ceil((last_row - first_row) / 3);
    left_lung_increment = ceil((last_row - first_row) / 2);

    temp = curr_mask;
    temp (temp < 2) = 0;
    temp(temp == 2) = 1;
    LLL = nnz(temp(first_row + left_lung_increment:last_row,:) .* Preprocessed_Image(first_row + left_lung_increment:last_row,:)) /(nnz(temp(first_row + left_lung_increment:last_row,:)));
    LUL = nnz(temp(first_row :first_row + left_lung_increment,:) .* Preprocessed_Image(first_row :first_row + left_lung_increment,:)) / (nnz(temp(first_row :first_row + left_lung_increment,:)));
    
    temp = curr_mask;
    temp(temp > 1) = 0; 
    RLL = nnz(temp(last_row - right_lung_increment:last_row,:) .* Preprocessed_Image(last_row - right_lung_increment:last_row,:)) / (nnz(temp(last_row - right_lung_increment:last_row,:))); 
    RUL = nnz(temp(first_row :first_row + right_lung_increment,:) .* Preprocessed_Image(first_row :first_row + right_lung_increment,:)) / (nnz(temp(first_row :first_row + right_lung_increment,:)));
    RML = nnz(temp(first_row + right_lung_increment :last_row - right_lung_increment,:) .* Preprocessed_Image(first_row + right_lung_increment :last_row - right_lung_increment,:)) / (nnz(temp(first_row + right_lung_increment :last_row - right_lung_increment,:)));
    
    severity = Find_LobeScore(LLL) + Find_LobeScore(LUL) + Find_LobeScore(RLL) + Find_LobeScore(RML) + Find_LobeScore(RUL);
%     x= (first_row+increment);
%     temp(x,:) = 1;
%     x = (first_row+(2*increment));
%     temp(x,:) = 1;
%     temp(last_row,:) = 1;
end

function lobeScore = Find_LobeScore(lobe)

    lobe(isnan(lobe))=0;
    if (ceil(lobe * 100) <= 5 & ceil(lobe * 100) > 0)
        lobeScore = 1;
    elseif(ceil(lobe * 100) > 5 & ceil(lobe * 100) <= 25)
        lobeScore =2;
    elseif(ceil(lobe * 100) > 25 & ceil(lobe * 100) <= 50)
        lobeScore =3;
    elseif(ceil(lobe * 100) > 50 & ceil(lobe * 100) <= 75)
        lobeScore =4;
    elseif(ceil(lobe * 100) > 75 & ceil(lobe * 100) <= 100)
        lobeScore = 5;
	else
    	lobeScore =0;
    end
end 

function[first_row, last_row] = Find_RowIndices(lung_mask)
    [~, columns] = size(lung_mask);
    lung_mask(lung_mask > 1) = 1;
    B = zeros(2, columns);
    for col = 1 : size(lung_mask, 2)
        if (isempty(find(lung_mask(:, col), 1, 'first')))
             B(1, col) = 513;   
        else
             B(1, col) = find(lung_mask(:, col), 1, 'first');
        end
        if (isempty(find(lung_mask(:, col), 1, 'last')))
             B(2, col) = 0; 
        else
             B(2, col) = find(lung_mask(:, col), 1, 'last');
        end
	   
    end
    first_row = min(B(1,:));
    last_row = max(B(2,:));
end
