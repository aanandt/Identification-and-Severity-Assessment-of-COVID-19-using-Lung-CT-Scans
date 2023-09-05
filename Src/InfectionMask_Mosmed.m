function[] = InfectionMask_Mosmed(img_filename, mask_seg_filename, mask_gt_filename, dest_dir)

    Mask_GT_files = dir(fullfile(mask_gt_filename));
    Patientwise_dice_score = {};
    pat_index = 1;
    all_count =1;
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
    similarity = [];
    for i = 3: 1: numel(Mask_GT_files)

        patient_name = Mask_GT_files(i).name
        
        filename = [patient_name(1:numel(Mask_GT_files(i).name) - 9), '.nii']; 
        Mask_ = niftiread(fullfile(mask_gt_filename, patient_name));
        [row, col, len] = size(Mask_);

        imagepath = fullfile(mask_seg_filename, filename, 'image.mat');
        mask_seg_path = fullfile(mask_seg_filename, filename, 'mask.mat');
        
        for index = 1: 1: len
            
            gt_mask = Mask_(:,:,index);
            Mask= load(mask_seg_path).data();
            curr_mask = squeeze(double(Mask(index, : , :)));
            curr_mask(curr_mask > 1) = 1;
            flag = 0;
            lung_area = nnz(curr_mask);
            total_area = 512 * 512;

            lung_involvement = ((lung_area * 100)/total_area);
            if lung_involvement >= 7

                flag = 1;
            end
            start_ = 1;
            end_ = len;
            
            if ((sum(sum(gt_mask)) >= 0)  & index >= start_ & index <= end_ & flag)
                

                Infection_Mask = InfectionMask_Generation(imagepath, mask_seg_path, index);
                
                
                rot_mask = imrotate(Infection_Mask, 90);
                Infection_Mask = flipdim(rot_mask, 1);
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
                dest_dir_4 = fullfile(dest_dir, 'Original_Mask', patient_name);
                if ~exist(dest_dir_4, 'dir')
                    mkdir(dest_dir_4)
                end
                
                filename_  = [dest_dir_1,'/', patient_name, '_Pred_mask_',num2str(index),'.png'];
                imwrite(Infection_Mask, filename_)
                
                filename_  = [dest_dir_2,'/', patient_name, '_GT_mask_',num2str(index),'.png'];
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

                filename_  = [dest_dir_4,'/lung_mask_',num2str(index),'.png'];
                imwrite(double(curr_mask), filename_)
                gt_mask = imresize(gt_mask, [512, 512]); 
                [Accuracy, Sensitivity, Fmeasure, Precision, MCC, Dice, Jaccard, Specificity] = EvaluateImageSegmentationScores(logical(gt_mask), logical(Infection_Mask));
                
                dice_score(mask_counter) = Dice;
                if (sum(sum(gt_mask ))== 0 & sum(sum(Infection_Mask))== 0)
                    % // Accuracy_score(mask_counter) = 1;
                    % // Sensitivity_score(mask_counter) = 1;
                    % // Fmeasure_score(mask_counter) = 1;
                    % // Precision_score(mask_counter) = 1;
                    % // MCC_score(mask_counter) = 1;
                    % // My_dice_score(mask_counter) = 1;
                    % // Jaccard_score(mask_counter) = 1;
                    % // Specificity_score(mask_counter) = 1;
                    % // mask_counter = mask_counter + 1;
                    disp('Hi')
                end
                if (sum(sum(gt_mask )) > 0 & sum(sum(Infection_Mask)) ~= 0)
                    
                    
                    Accuracy_score(mask_counter) = Accuracy;
                    Sensitivity_score(mask_counter) = Sensitivity;
                    Fmeasure_score(mask_counter) = Fmeasure;
                    Precision_score(mask_counter) = Precision;
                    MCC_score(mask_counter) = MCC;
                    My_dice_score(mask_counter) = Dice;
                    Jaccard_score(mask_counter) = Jaccard;
                    Specificity_score(mask_counter) = Specificity;
                    mask_counter = mask_counter + 1;
                end
                    
                    
                    all_count = all_count + 1;
                    
                
            end
        end
        if (mask_counter ~= 1)
            patient_score = ((sum(dice_score))*100)/(mask_counter-1);
            Patientwise_dice_score{pat_index, 1} = patient_name;
            Patientwise_dice_score{pat_index, 2} = patient_score;
            Patientwise_dice_score{pat_index, 3} = dice_score(:);
            Patients_scores{pat_index, 1} = ((sum(Accuracy_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 2} = ((sum(Sensitivity_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 3} = ((sum(Precision_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 4} = ((sum(MCC_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 5} = ((sum(Jaccard_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 6} = ((sum(Specificity_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 7} = ((sum(My_dice_score))*100)/(mask_counter-1);
            Patients_scores{pat_index, 8} = ((sum(Fmeasure_score))*100)/(mask_counter-1);

            pat_index = pat_index + 1;
        end
        
    end
    % disp('Sensitivity_score')  
    % mean(Sensitivity_score)
    % disp('dice_score')
    % mean(My_dice_score)
    % disp('Specificity_score')
    % mean(Specificity_score)
    % disp('Precision_score')
    % mean(Precision_score)
    if (pat_index >= 1)
        temp = Patientwise_dice_score(:,2);
        all_scores = [temp{:}];
        Accuracy = mean(all_scores)

        temp = Patients_scores(:,1);
        all_scores = [temp{:}];
        Accuracy = mean(all_scores)

        temp = Patients_scores(:,2);
        all_scores = [temp{:}];
        Sensitivity = mean(all_scores)

        temp = Patients_scores(:,3);
        all_scores = [temp{:}];
        Precision = mean(all_scores)

        temp = Patients_scores(:,4);
        all_scores = [temp{:}];
        MCC_score = mean(all_scores)

        temp = Patients_scores(:,5);
        all_scores = [temp{:}];
        Jaccar = mean(all_scores)

        temp = Patients_scores(:,6);
        all_scores = [temp{:}];
        Specificity = mean(all_scores)
        
        temp = Patients_scores(:,7);
        all_scores = [temp{:}];
        My_dice_score = mean(all_scores)

        temp = Patients_scores(:,8);
        all_scores = [temp{:}];
        Fmeasure = mean(all_scores)


    end
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
    Sensitivity = TP/(TP+FN);
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
%{
function[Infection_mask] = InfectionMask_Generation_Mosmed(Imagepath, Maskpath, index)
    
    niftifiles = load(Imagepath).data();%niftiread(fullfile(filename));%
    Mask= load(Maskpath).data();
    Image = double(squeeze(niftifiles(index,:,:)));
    curr_mask = squeeze(double(Mask(index, : , :)));
    curr_mask(curr_mask > 1) = 1;
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
        GMModel = fitgmdist(S,3);
        %GMModel = fitgmdist(S,3,'RegularizationValue',0.1);
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
    final_mask = bwareaopen(final_mask, 70);
    se = strel('disk', 2);
    final_mask = imfill(final_mask, 'holes');

    segmented_img = final_mask .* seg_Image;
    
    
    
    Infection_mask = segmented_img;
    segmented_img(segmented_img == 0)=-1024;
    Infection_mask(Infection_mask == -1024) = 0;
    Infection_mask(Infection_mask ~= 0) = 1;        
    se = strel('disk', 2);
    Infection_mask = imdilate(Infection_mask,se);
end

%}