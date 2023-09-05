function [] = PreprocessPipeline_Mosmed(inputDir, Mask_path, outputDir)

	Preprocessing(inputDir, outputDir, Mask_path);
	
end

function Preprocessing(folder_path, outputDir, Mask_path)
	folder_name = fullfile(folder_path);
	patients = dir(fullfile(folder_path));
	patients(1:2) = [];
	delete(gcp('nocreate'));
	parpool(16)
	parfor pat_index = 1 : 1 : numel(patients)

		patients_name = patients(pat_index).name
		filename = fullfile(Mask_path, patients_name, 'image.mat');
		dicomfiles = load(filename).data();
		[len, row, col] = size(dicomfiles);
		
		mask_filename = fullfile(Mask_path, patients_name, 'mask.mat');
		Mask = load(mask_filename).data();
		ser_index = 1;
		for slice_index = 1 : 1: len

			
			Image = squeeze(double(dicomfiles(slice_index, : , :)));
			Image = imrotate(Image, 180);
		    %min(min(Image))
			%%%%%%%%%%% Generating Mask %%%%%%%%%%%%%%%%%
			
			curr_mask = squeeze(double(Mask(slice_index, : , :)));
			curr_mask = imrotate(curr_mask, 180);
			curr_mask(curr_mask>1) = 1;
			seg_Image = Image .* curr_mask;
			
			flag = 0;
	    	lung_area = nnz(curr_mask);
	    	total_area = 512 * 512;
	    	lung_involvement = ((lung_area * 100)/total_area);
	    	if lung_involvement >= 7

	    		flag = 1;
	    	end
	    	start_ = 10;
	    	end_ = len-5;

			if ((~ all(seg_Image(:) < 2) & flag == 1) & ~(slice_index < start_) & ~(slice_index > end_))
				
				x = ['Preprocessing strated for patient:', patients_name,  '\tslice number:', num2str(slice_index)];
				disp(x)

				%%%%%%%%%% Fit GMM on Histogram of Intensities%%%%%%%%%%%%%%%%%
				S = reshape(seg_Image,[],1);
				max_val = max(max(seg_Image));
				min_val = min(min(seg_Image));
				num_bins = ceil((max_val - min_val)/3);
				S = S(S ~= 0);
				
				try
    				GMModel = fitgmdist(S,3);
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

					final_mask = ceil((new_img + new_img_1) / 2);
					se = strel('disk', 4);
					final_mask_1 = imfill(final_mask, 'holes');
					%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

					seg_Image = final_mask_1 .* seg_Image;
					curr_mask_prewitt = edge(curr_mask,'Prewitt');
					curr_mask_roberts = edge(curr_mask,'roberts');
					curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2)* -100);
					
					
					volume = seg_Image + curr_mask_final;
					volume(volume==0)=-1024;
					min_ = min(min(volume));
					max_ = max(max(volume));
					volume(volume < min_) = min_;
					volume(volume > max_) = max_;
					volume = (volume - min_) / (max_ - min_);
					volume = uint8(volume.*255);
					
					
			    	dest_dir = fullfile(outputDir, patients_name);
			    	if ~exist(dest_dir, 'dir')
			    		mkdir(dest_dir)
			        end
					fname = [dest_dir, '/', patients_name, '_', num2str(slice_index),'.jpg'];
					
					parsave(fname, volume)
				catch
    				warning('Ill-conditioned covariance created in processing');
    				
				end
			end
			
		end
	
	end
		
end

function [subFolders] = Read_folders(InputDir)
	files = dir(fullfile(InputDir));
	dirFlags = [files.isdir];
	subFolders = files(dirFlags);
	subFolders(1:2) = [];
end

function parsave(fname, x)
	
  	imwrite(x, fname);
  	
end


