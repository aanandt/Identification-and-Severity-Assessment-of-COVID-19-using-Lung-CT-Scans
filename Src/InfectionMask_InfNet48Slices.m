function InfectionMask_InfNet48Slices(img_filename, mask_seg_filename, mask_gt_filename, dest_dir)


    %%%% 48 slcies
    %img_filename = '../TempDir/Matlab_Mask_files_Test_InfiNetData_48Slices/';
    %mask_seg_filename = '../TempDir/Matlab_Mask_files_Test_InfiNetData_48Slices/';
    %mask_gt_filename = '../../Dataset/InfNet/Mask/';

    Mask_GT_files = dir(fullfile(mask_gt_filename));
    all_count = 1;
    mask_counter = 1;
    dice_score = [];
    Accuracy_score = [];
    Precision_score = [];
    Sensitivity_score =[];
    Specificity_score = [];
    MCC_score = [];
    Jaccard_score =[];
    My_dice_score = [];
    Fmeasure_score = [];

    MAE = [];
    for i = 3 : 1: 3%numel(Mask_GT_files)

        patient_name = 'tr_mask.nii.gz';
        slice_indices = [7,8,9,10,19,22,37,39,60,61,62,63,64,65,66,67,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]; %%% Medseg Dataset 48 slices
        filename =  'tr_im.nii.gz'
        
        Mask_ = niftiread(fullfile(mask_gt_filename, patient_name));%niftiread(fullfile(mask_gt_filename));%
        [row, col, len] = size(Mask_);

        imagepath = fullfile(img_filename, filename, 'image.mat');
        mask_seg_path = fullfile(mask_seg_filename, filename, 'mask.mat');
            
            for ind = 1 : 1 : length(slice_indices)
                index = slice_indices(ind);

                gt_mask = Mask_(:,:,index);
                 
                onethird = ceil((len/3));
                twothird = len - onethird;

                Mask= load(mask_seg_path).data();
                curr_mask = squeeze(double(Mask(index, : , :)));

                
                start_ = 1;
                end_ = len;
                
                if ( index >= start_ & index <= end_)  
                    

                    Infection_mask = InfectionMask_Generation(imagepath, mask_seg_path, index);
                    disp(index);
                    
                    
                    rot_mask = imrotate(Infection_mask, 90);
                    Infection_mask = flipdim(rot_mask, 1);

                    gt_mask(gt_mask > 1) = 1;
                    
                    
                    dest_dir_1 = fullfile(dest_dir, 'Predicted_Mask');
                    if ~exist(dest_dir_1, 'dir')
                        mkdir(dest_dir_1)
                    end
                    
                    dest_dir_2 = fullfile(dest_dir, 'GT_Mask');
                    if ~exist(dest_dir_2, 'dir')
                        mkdir(dest_dir_2)
                    end
                    
                    dest_dir_3 = fullfile(dest_dir, 'Original_Image', patient_name);
                    if ~exist(dest_dir_3, 'dir')
                        mkdir(dest_dir_3)
                    end

                    filename_  = [dest_dir_1,'/', num2str(i-2), '_',num2str(index-1),'.png'];
                    imwrite(Infection_mask, filename_)
                    
                    filename_  = [dest_dir_2,'/', num2str(i-2), '_',num2str(index-1),'.png'];
                    imwrite(double(gt_mask), filename_)
                    filename_  = [dest_dir_3,'/image_',num2str(index),'.png'];
                    niftifiles = load(imagepath).data();%niftiread(fullfile(filename));%
        
                    volume = double(squeeze(niftifiles(index,:,:)));
                    min_ = min(min(volume));
                    max_ = max(max(volume));
                    volume(volume < min_) = min_;
                    volume(volume > max_) = max_;
                    volume = (volume - min_) / (max_ - min_);
                    volume = uint8(volume.*255);
                    imwrite((volume), filename_)
                    
                    gt_mask = imresize(gt_mask, [512, 512]);

                    [Accuracy, Sensitivity, Fmeasure, Precision, MCC, Dice, Jaccard, Specificity] = EvaluateImageSegmentationScores(logical(gt_mask), logical(Infection_mask));
                    
                    
                    mae = CalMAE(logical(Infection_mask), logical(gt_mask));
                    
                                  
                    dice_score(mask_counter) = Dice;
                    if (sum(sum(gt_mask ))== 0 & sum(sum(Infection_mask))== 0)
                        % Accuracy_score(mask_counter) = 1;
                        % Sensitivity_score(mask_counter) = 1;
                        % Fmeasure_score(mask_counter) = 1;
                        % Precision_score(mask_counter) = 1;
                        % MCC_score(mask_counter) = 1;
                        % My_dice_score(mask_counter) = 1;
                        % Jaccard_score(mask_counter) = 1;
                        % Specificity_score(mask_counter) = 1;
                        
                        % MAE(mask_counter) = mae;
                        
                        % mask_counter = mask_counter + 1;
                        disp('Hi')
                    end
                    if (sum(sum(gt_mask )) >= 0 & sum(sum(Infection_mask)) ~= 0)
                        
                        
                        Accuracy_score(mask_counter) = Accuracy;
                        Sensitivity_score(mask_counter) = Sensitivity;
                        Fmeasure_score(mask_counter) = Fmeasure;
                        Precision_score(mask_counter) = Precision;
                        MCC_score(mask_counter) = MCC;
                        My_dice_score(mask_counter) = Dice;
                        Jaccard_score(mask_counter) = Jaccard;
                        Specificity_score(mask_counter) = Specificity;
                        
                        MAE(mask_counter) = mae;
                      

                        mask_counter = mask_counter + 1;
                    end
                    

                        
                    
                    all_count = all_count + 1;
                end
                
            end
            
    end
    disp('Sensitivity_score')  
    mean(Sensitivity_score)
    disp('dice_score')
    mean(My_dice_score)
    disp('Specificity_score')
    mean(Specificity_score)
    disp('Precision_score')
    mean(Precision_score)
end

function [Accuracy, Sensitivity, Fmeasure, Precision, MCC, Dice, Jaccard, Specitivity] = EvaluateImageSegmentationScores(A, B)
    
    % A and B need to be binary images
    % A is the ground truth, B is the segmented result.
    % MCC - Matthews correlation coefficient
    % Note: Sensitivity = Recall
    % TP - true positive, FP - false positive, 
    % TN - true negative, FN - false negative
    
    % If A, B are binary images, but uint8 (0, 255),
    % Need to convert to logical images.
    if(isa(A,'logical'))
        X = A;
    else
        X = imbinarize(A);
    end
    if(isa(B,'logical'))
        Y = B;
    else
        Y = imbinarize(B);
    end
    
    % Evaluate TP, TN, FP, FN
    sumindex = X + Y;
    TP = length(find(sumindex == 2));
    TN = length(find(sumindex == 0));
    substractindex = X - Y;
    FP = length(find(substractindex == -1));
    FN = length(find(substractindex == 1));
    Accuracy = (TP+TN)/(FN+FP+TP+TN);
    if (TP + FN == 0)
        Sensitivity = 0;
    else
        Sensitivity = TP/(TP+FN);
    end
    Precision = TP/(TP+FP);
    Fmeasure = 2*TP/(2*TP+FP+FN);
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    
    % If you use MATLAB2017b+, you can call: Dice = dice(A, B), but A, B
    % need to be converted to the logical form
    % If you use MATLAB2017b+, you can call: Jaccard = jaccard(A, B), but
    % A, B need to be converted to the logical form
    Dice = 2*TP/(2*TP+FP+FN);
    Jaccard = Dice/(2-Dice);
    Specitivity = TN/(TN+FP);
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


function mae = CalMAE(smap, gtImg)
    % Code Author: Wangjiang Zhu
    % Email: wangjiang88119@gmail.com
    % Date: 3/24/2014
    if size(smap, 1) ~= size(gtImg, 1) || size(smap, 2) ~= size(gtImg, 2)
        error('Saliency map and gt Image have different sizes!\n');
    end

    if ~islogical(gtImg)
        gtImg = gtImg(:,:,1) > 128;
    end

    smap = im2double(smap(:,:,1));
    fgPixels = smap(gtImg);
    fgErrSum = length(fgPixels) - sum(fgPixels);
    bgErrSum = sum(smap(~gtImg));
    mae = (fgErrSum + bgErrSum) / numel(gtImg);
end
