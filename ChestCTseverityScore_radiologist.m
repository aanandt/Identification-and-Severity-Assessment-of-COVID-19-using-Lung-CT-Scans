clear all;
close all;
clc;

input_path = 'SeverityData/'
outputDir='Dump/SeverityData/'
Mask_path = 'SeverityData/';

patients = Read_folders(input_path);
patientswise_CTSS = {};
patients_list = [];
for pat_index = 1 : 1 : numel(patients)
	patients_name = patients(pat_index).name;
	%if(strcmp(patients_name, 'T1-024'))

		masks = load(fullfile(input_path, patients_name, 'mask.mat')).data();
		CTscans = load(fullfile(input_path, patients_name, 'image.mat')).data();
		[len, row, col] = size(masks);

		SeverityScore=[];
		index = 1;
		[first_row, last_row] = Find_MinMaxRowIndices(masks);

		for slice_index = 1: 1 : len
		    
		    curr_mask =  squeeze(double(masks(slice_index,:,:)));
		    Image = squeeze(double(CTscans(len-slice_index+1,:,:)));
		    lung_area = nnz(curr_mask);
			total_area = 512 * 512;
		   
			lung_involvement = ((lung_area * 100)/total_area);
			start_ = 7;
			end_ = len-5;
			if ((lung_involvement >= 6) & (~(slice_index < start_) & ~(slice_index > end_)))
		        
		        temp = curr_mask;
		        temp(temp>1) = 1;
		        
		        PreprocessedImage = Preprocessing(temp, Image, len, slice_index);
		        PreprocessedImage = double(PreprocessedImage);
				%[first_row, last_row] = Find_RowIndices(curr_mask);     
		        Severity_Score(index) = Find_SeverityScore(curr_mask, first_row, last_row, double(PreprocessedImage));
		        
		        index = index + 1;
		    end
		    
		end
		
		patientswise_CTSS{pat_index,1} = patients_name;
		patientswise_CTSS{pat_index,2} = mean(Severity_Score);
	%end
end

function[volume] =  Preprocessing(curr_mask, Image, len, slice_index)
	
    %curr_mask = squeeze(double(Mask(slice_index, : , :)));
    hounsfieldImage = Get_hounsfieldImage(Image);
	seg_Image = hounsfieldImage .* curr_mask;
	
	

	if (~ all(seg_Image(:) < -1000))
		
		%%%%%%%%%% Fit GMM on Histogram of Intensities%%%%%%%%%%%%%%%%%
		S = reshape(seg_Image,[],1);
		max_val = max(max(seg_Image));
		min_val = min(min(seg_Image));
		num_bins = ceil((max_val - min_val)/3);
		S = S(S ~= 0);
		
		try
			GMModel = fitgmdist(S,3);
		catch
 			warning('Ill-conditioned covariance created in processing');
 			volume = seg_Image;
 		end	
			img_reco = reshape(seg_Image,[],1);
			clusterX = cluster(GMModel,img_reco);
			[mean, index] = sort(GMModel.mu(:));
			std = sort(GMModel.Sigma(:));
            
			%%%%%%%%%%%%% Adaptive thresholding with GMM %%%%%%%%%%%%%%%%%%%
			new_img = seg_Image;
			lower_val = (mean(2)- (1.5 * sqrt(GMModel.Sigma(index(2)))));
			upper_val = (mean(2) + (1.5 * sqrt(GMModel.Sigma(index(2)))));
			if upper_val <= 30
			    new_img(new_img >= upper_val) = 0;
			else 
			    new_img(new_img >= 10) = 0;
			end
			if lower_val >= -650
			    new_img(new_img <= lower_val) = 0;
			else 
			    new_img(new_img <= -650) = 0;
			end
			new_img(new_img ~= 0 ) = 1;

			%%%%%%%%%%%%% Adaptive thresholding with GMM %%%%%%%%%%%%%%%%%%%
			new_img_1 = seg_Image;
			lower_val = (mean(3)- (1.5 * sqrt(GMModel.Sigma(index(3)))));
			upper_val = (mean(3) + (1.5 * sqrt(GMModel.Sigma(index(3)))));
			if upper_val <= 30
			    new_img_1(new_img_1 >= upper_val) = 0;
			else 
			    new_img_1(new_img_1 >= 30) = 0;
			end
			if lower_val >= -650
			    new_img_1(new_img_1 <= lower_val) = 0;
			else 
			    new_img_1(new_img_1 <= -650) = 0;
			end
			new_img_1(new_img_1 ~= 0 ) = 1;
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
			volume = Infection_mask;
			min_ = min(min(volume));
			max_ = max(max(volume));
			volume(volume < min_) = min_;
			volume(volume > max_) = max_;
			volume = (volume - min_) / (max_ - min_);
			volume = uint8(volume.*255);
		
% 		    	dest_dir = fullfile(outputDir, patients_name);
% 		    	if ~exist(dest_dir, 'dir')
% 		    		mkdir(dest_dir)
% 		        end
% 				fname = [dest_dir, '/', patients_name, '_', num2str(slice_index),'.jpg'];
% 				
% 				save(fname, volume)
 		
    end
		
end

function [subFolders] = Read_folders(InputDir)
	files = dir(fullfile(InputDir));
	dirFlags = [files.isdir];
	subFolders = files(dirFlags);
	subFolders(1:2) = [];
end
function [hounsfieldImage]= Get_hounsfieldImage(Image)

	rSlope = 1;%info.RescaleSlope;
	rinter = -1024;%info.RescaleIntercept;
	for j = 1 : size(Image, 1) % This loop multiply each voxel value by the rescale slope
	    for i = 1 : size(Image, 2)
	        hounsfieldImage(i,j) = Image(i,j)*rSlope + rinter;
	    end
	end
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

% function parsave(fname, x)
% 	
%   	imwrite(x, fname);
%   	
% end


