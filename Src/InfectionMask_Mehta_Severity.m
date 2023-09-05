clear all;
close all;
clc;

img_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mehta/';
mask_seg_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/TempDir/Matlab_Mask_files_Test_Mehta/';
mask_gt_filename = '/speech/tmp/anand/Covid_Dataset/MY_TestData/COVID_MEHTA/Dataset/';



Mask_GT_files = dir(fullfile(mask_gt_filename));
Patientwise_volume_score = {};
Patients_scores = {};
all_count = 1;
pat_index = 1;
medseg_index = 1;
for i = 3 : 1: numel(Mask_GT_files)

    patient_name = Mask_GT_files(i).name;
    filename =  patient_name
    imagepath = fullfile(img_filename, filename, 'image.mat');
    mask_seg_path = fullfile(mask_seg_filename, filename, 'mask.mat');
    

    Mask= load(mask_seg_path).data();
    [len, row, col] = size(Mask);
    mask_counter = 1;
    severity = [];
    for index = 1:1:len
       
        
        curr_mask = squeeze(double(Mask(index, : , :)));
        flag = 0;
        mid = floor(len/2);
        if( len > 100)
            start_ = mid -50 ;
            end_ = mid + 50;
            flag = 1;
        else
            start_ = index;
            end_ = len;
            lung_area = nnz(curr_mask);
            total_area = 512 * 512;
            lung_involvement = ((lung_area * 100)/total_area);
            if lung_involvement >= 10

                flag = 1;
            end

        end
        
        if ( index >= start_ & index <= end_& flag )  
            

            Infection_Mask = Generate_Infection_Mask(imagepath, mask_seg_path, index);%curr_mask;%
            rot_mask = imrotate(Infection_Mask, 90);
            Infection_Mask = flipdim(rot_mask, 1);
            dest_dir_1 = fullfile('Dump/Mask_Dicescore_Mehta/', patient_name,'/Predicted_Mask');
            
            if ~exist(dest_dir_1, 'dir')
                mkdir(dest_dir_1)
            end
            
            dest_dir_3 = fullfile('Dump/Mask_Dicescore_Mehta/', patient_name,'/Original_Image');
            if ~exist(dest_dir_3, 'dir')
                mkdir(dest_dir_3)
            end

            filename_  = [dest_dir_1,'/Infection_Mask_',num2str(index),'.png'];
            imwrite(Infection_Mask, filename_)
            
            filename_  = [dest_dir_3,'/image_',num2str(index),'.png'];
            niftifiles = load(imagepath).data();
            volume = double(squeeze(niftifiles(index,:,:)));
            min_ = min(min(volume));
            max_ = max(max(volume));
            volume(volume < min_) = min_;
            volume(volume > max_) = max_;
            volume = (volume - min_) / (max_ - min_);
            volume = uint8(volume.*255);
            imwrite((volume), filename_)
            
            severity(mask_counter) = sum(sum(Infection_Mask)) / sum(sum(curr_mask));
            all_count = all_count + 1;
            mask_counter = mask_counter + 1;
        end
        
    end
    patient_score = ((sum(severity))*100)/(mask_counter-1);
    Patientwise_volume_score{pat_index, 1} = patient_name;
    Patientwise_volume_score{pat_index, 2} = patient_score;
    pat_index = pat_index + 1;

end


function[Infection_Mask] = Generate_Infection_Mask(Imagepath, Maskpath, index)
    
    niftifiles = load(Imagepath).data();%niftiread(fullfile(filename));%
    Mask= load(Maskpath).data();
    
    
    Image = double(squeeze(niftifiles(index,:,:)));
    rSlope = 1; %info.RescaleSlope;
    rinter = -1024; %info.RescaleIntercept;
    for j = 1 : size(Image, 1) % This loop multiply each voxel value by the rescale slope
        for i = 1 : size(Image, 2)
            hounsfieldImage(i,j) = Image(i,j)*rSlope + rinter;
        end
    end

    curr_mask = squeeze(double(Mask(index, : , :)));
    Image(Image <= -1024)= -1024;
    Image(Image >= 300)= 300;
    
    seg_Image = hounsfieldImage .* curr_mask;
    org_seg_Image = Image .* curr_mask;

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
    
    V2 = vesselness2D((seg_Image), 0.5:0.5:2.5, [1;1], 1, true);
    V2(V2 < 0.65) = 0;
    V2(V2 ~= 0) = 1;
    
    [row, col] = size(final_mask);
    final_mask_1 = final_mask;
    for i = 1 : 1: row
        for j = 1 : 1: col
            if V1(i,j) == 1
                final_mask(i,j) == 0;
            end
        end
    end
    
    seg_Image(seg_Image ==0)=-1024;
    segmented_img(segmented_img ==0)=-1024;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    curr_mask_prewitt = edge(curr_mask,'Prewitt');
    curr_mask_roberts = edge(curr_mask,'roberts');
    curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2));
    
    for j = 1 : size(curr_mask_final, 1) % This loop multiply each voxel value by the rescale slope
        for i = 1 : size(curr_mask_final, 2)
            if (curr_mask_final(i,j) == 250)
                segmented_img(i,j) = curr_mask_final(i,j);
            end
            
        end
    end
    
    
    %%%%%%%%%%%Windowing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    med_window_img = Windowing(seg_Image, 350, 50);
    med_window = med_window_img;
    
    med_window(med_window ~= -125) = 1;
    med_window(med_window == -125) = 0;
    med_window_mask = bwareaopen(med_window, 70);
    %img= med_window .* (org_seg_Image);
    final_mask = ceil((final_mask + med_window_mask) / 2);
    %final_mask = ceil((final_mask + curr_mask_final) / 2);
    
    
    my_img= final_mask .* seg_Image;%final_mask .* segmented_img;%
    my_img(my_img==0)=-1024;
    volume = my_img;
    min_ = -1150;
    max_ = 150;
    volume(volume < min_) = min_;
    volume(volume > max_) = max_;
    volume = (volume - min_) / (max_ - min_);
    volume = uint8(volume.*255);
    Infection_Mask = my_img;
    Infection_Mask(Infection_Mask == -1024) = 0;
    Infection_Mask(Infection_Mask ~= 0) = 1;        
    se = strel('disk', 4);
    Infection_Mask = imdilate(Infection_Mask,se);
   
end
function [severity_score] = find_severity(final_processed_img, curr_mask)

    lung_area = nnz(curr_mask);
    infected_area = nnz(final_processed_img);
    severity_score = infected_area / lung_area;
    
end
function [window_img] = Windowing(Image, WW, WL)

    x = WL + (WW / 2);
    y = WL - (WW / 2);
    window_img = Image;
    for j = 1 : size(Image, 1) % This loop multiply each voxel value by the rescale slope
        for i = 1 : size(Image, 2)
            if (Image(i,j) < y)
                window_img(i,j) = y;
            end
            if (Image(i,j) > x)
                window_img(i,j) = x;
            end
        end
    end
end

