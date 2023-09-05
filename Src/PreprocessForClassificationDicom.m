function [] = preprocessing()

	inputDir = '../Dataset/SPGC/'
	outputDir = '../Preprocessed_Datasets/MY_JPG/'
	%input_file = '../Dataset/Slice_level_labels.csv';
	Mask_path = '../TempDir/Matlab_Mask_files_Test/';
	input_file = '../Dataset/Slice_level_labels_dummy.csv';
	
	subFolders = Read_folders(inputDir);
	for i = 1 : 1 : numel(subFolders)

		subFolder = subFolders(i).name;
		if (strcmp( subFolder, 'Train'))

			folder_path = fullfile(inputDir, subFolder);
			categories = Read_folders(folder_path);
			for j = 1 : 1 : numel(categories)
				
				category = categories(j).name;
				folder_path = fullfile(inputDir, subFolder, category);
				subsets = Read_folders(folder_path);
				
				for k = 1 : 1 : numel(subsets)
					
					subset = subsets(k).name;
					folder_path = fullfile(inputDir, subFolder, category, subset);
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, outputDir, Mask_path);
			
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

function [all_patient_label] = Get_slice_level_label(input_file)

	fileID = fopen(input_file);
	data = fgetl(fileID);
	patient_label = [];
	all_patient_label = {};
	index = 1;
	while ~feof(fileID)
		data = fgetl(fileID);
		sdata = split(data,',');
		patient_label = str2double(sdata(2:end));		
	    all_patient_label{index,1} = sdata{1};
	    all_patient_label{index,2} = patient_label;
	    index = index + 1;
	end
end

function [Severity_score_patient] = Extract_Dicom_Preprocessing(folder_path, category, subFolder, outputDir, Mask_path)
	
	folder_name = fullfile(folder_path);
	patients = Read_folders(fullfile(folder_path))
	delete(gcp('nocreate'));
	parpool(8)
	parfor pat_index = 1 : 1 : numel(patients)

		patients_name = patients(pat_index).name;
		filename = fullfile(folder_name, patients_name)
		dicomfiles_temp = dir(fullfile(filename, '*.dcm'));
		xlsfiles = {dicomfiles_temp.name};
	    [~,idx] = natsortfiles(xlsfiles);
	    dicomfiles = dicomfiles_temp(idx);
		%disp(dicomfiles)

		for slice_index = 1 : 1: numel(dicomfiles)

			dicom_file = dicomfiles(slice_index).name;
			Image = dicomread(fullfile(filename,dicom_file));
		    info = dicominfo(fullfile(filename,dicom_file));

			mask_filename = fullfile(Mask_path, category, patients_name, 'mask.mat');
		    Mask = load(mask_filename).data();
		    
		    Image = double(Image);
		    fullfile(filename,dicom_file);
			rSlope = info.RescaleSlope;
			rinter = info.RescaleIntercept;
			hounsfieldImage = Get_hounsfieldImage(Image, info);
			
			%%%%%%%%%%% Generating Mask %%%%%%%%%%%%%%%%%

			curr_mask = squeeze(double(Mask(slice_index, : , :)));
			seg_Image = hounsfieldImage .* curr_mask;
			org_seg_Image = Image .* curr_mask;

			mid = floor((numel(dicomfiles)/2));
			   
			flag =1;
			if( numel(dicomfiles) > 80)
	    		start_ = mid -40 ;
	    		end_ = mid + 40;
			
			else
	    		start_ = mid - 25;
	    		end_ = mid + 25;
	    	end

			if ((~ all(seg_Image(:) < 2) & flag == 1) & ~(info.InstanceNumber < start_) & ~(info.InstanceNumber > end_))%(numel(dicomfiles) - 5)
				
				dicom_file = dicomfiles(slice_index).name
				x = ['Preprocessing strated for patient:', patients_name,  '\tslice number:', num2str(info.InstanceNumber)];
				disp(x)
				
				%%%%%%%%%% Fit GMM on Histogram of Intensities%%%%%%%%%%%%%%%%%
				S = reshape(seg_Image,[],1);
				max_val = max(max(seg_Image));
				min_val = min(min(seg_Image));
				num_bins = ceil((max_val - min_val)/3);
				S = S(S ~= 0);
				GMModel = fitgmdist(S,3);
				img_reco = reshape(seg_Image,[],1);
				clusterX = cluster(GMModel,img_reco);
				[mean, index] = sort(GMModel.mu(:));
				std = sort(GMModel.Sigma(:));

				%{
				gmm_x = linspace(min_val,max_val,num_bins);
				gmm_x = reshape(gmm_x, [], 1);
				gmm_y = pdf(GMModel, gmm_x)%@(gmm_x) arrayfun(@(x0) pdf(GMModel,[x0]),x);
				f = figure
				hist = histogram(S, num_bins, 'Normalization','pdf');
				hold on
				plot(gmm_x, gmm_y,"Color",'r')
				%}
				

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

				%%%%%%%%%%%%%%%% Fill the holes in the infection region %%%%%%%%%%%%%%
				final_mask = ceil((new_img + new_img_1) / 2);
				se = strel('disk', 4);
				final_mask_1 = imfill(final_mask, 'holes');		

				seg_Image = final_mask_1 .* seg_Image;
				org_seg_Image = final_mask_1 .* org_seg_Image;

				%%%%%%%%%%%%%%%Boundary generation from lung mask %%%%%%%%%%%%%%%%%%%
				
				curr_mask_prewitt = edge(curr_mask,'Prewitt');
				curr_mask_roberts = edge(curr_mask,'roberts');
				curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2) * 950);

				my_img= org_seg_Image + curr_mask_final;
				
				%%%%%%%%%%%%%%%% COnverting to grayscale image %%%%%%%%%%%%%%%%%%%
				volume = my_img;
				min_ = min(min(volume));
				max_ = max(max(volume));
				volume(volume < min_) = min_;
				volume(volume > max_) = max_;
				volume = (volume - min_) / (max_ - min_);
				volume = uint8(volume.*255);

				if strcmp(subFolder, 'Train') 
		    		dest_dir = fullfile(outputDir, subFolder, category, patients_name);
		    	else
		    		dest_dir = fullfile(outputDir, subFolder, category, patients_name);
		    	end
		    	if ~exist(dest_dir, 'dir')
		    		mkdir(dest_dir)
		        end
				fname = [dest_dir, '/', patients_name, '_', num2str(info.InstanceNumber),'.jpg'];
				
				parsave(fname, volume)
			end
		end
	
	end
		
end

function [hounsfieldImage]= Get_hounsfieldImage(Image, info)

	rSlope = info.RescaleSlope;
	rinter = info.RescaleIntercept;
	for j = 1 : size(Image, 1) % This loop multiply each voxel value by the rescale slope
	    for i = 1 : size(Image, 2)
	        hounsfieldImage(i,j) = Image(i,j)*rSlope + rinter;
	    end
	end
end

function parsave(fname, x)
	
  	imwrite(x, fname);
  	
end

function [window_img] = Windowing(Image, WW, WL)

    x = WL + (WW / 2);
    y = WL - (WW / 2);
    window_img = Image;
    for j = 1 : size(Image, 1) 
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

function [image] = Generate_mask_image(img, final_img)
    
    image = final_img;
    for j = 1 : size(img, 1) 
        for i = 1 : size(img, 2)
            if (img(i,j) == 750)
                image(i,j) = img(i,j);
            end
            
        end
    end
end


function [image] = Generate_final_image(img, final_img)
    
    image = img;
    for j = 1 : size(img, 1)
        for i = 1 : size(img, 2)
            if (img(i,j) ==0 )
                image(i,j) = final_img(i,j);
            end
            
        end
    end
end
