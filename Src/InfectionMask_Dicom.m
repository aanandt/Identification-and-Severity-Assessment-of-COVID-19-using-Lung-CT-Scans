function [] = InfectionMask_Dicom(inputDir, outputDir, Mask_path)

	%inputDir = '/speech/tmp/anand/Covid_Dataset/MY_TestData/COVID_LDCT/Dataset-S1_LDCT/'
	%outputDir = 'Preprocessed_Data_LDCT/'
	%input_file = '../Dataset/Slice_level_labels.csv';
	%Mask_path = 'TempDir/Matlab_Mask_files_Test_LDCT/';
	%input_file = '../Dataset/Slice_level_labels_dummy.csv';
	
	
	categories = Read_folders(inputDir);
	for i = 1 : 1 : numel(categories)
		category = categories(i).name;
		folder_path = fullfile(inputDir,category)
		%if not(isfolder(folder_path))
    	
		category = categories(i).name;
		folder_path = fullfile(inputDir,category);
		if category(1) == 'N'
			Extract_Dicom_Preprocessing(folder_path, outputDir, Mask_path);
		end
		if category(1) == 'C'
			Extract_Dicom_Preprocessing(folder_path, outputDir, Mask_path);
		end
		%end
	end
	
	
	%Extract_Dicom_Preprocessing(inputDir, outputDir, Mask_path);
	
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

function [Severity_score_patient] = Extract_Dicom_Preprocessing(folder_path, outputDir, Mask_path)
	folder_name = fullfile(folder_path)
	patients = dir(fullfile(folder_path))
	patients(1:2) = [];
	delete(gcp('nocreate'));
	parpool(16)
	parfor pat_index = 1: 1 : numel(patients)

		patients_name = patients(pat_index).name
		filename = fullfile(folder_name, patients_name);
		%{
		filename = fullfile(Mask_path, patients_name, 'image.mat');
		dicomfiles = load(filename).data();
		[len, row, col] = size(dicomfiles);
		%}
		dicomfiles_temp = dir(fullfile(filename, '*.dcm'));
		xlsfiles = {dicomfiles_temp.name};
        [~,idx] = natsortfiles(xlsfiles);
        dicomfiles = dicomfiles_temp(idx);
		disp(dicomfiles)
		
		mask_filename = fullfile(Mask_path, patients_name, 'mask.mat');
		Mask = load(mask_filename).data();
		for slice_index = 1 : 1: len

			%dicom_file = dicomfiles(slice_index).name;
			Image = squeeze(double(dicomfiles(slice_index, : , :)));
		    %info = dicominfo(fullfile(filename,dicom_file));
			%t = Tiff(fullfile(filename,dicom_file),'r');
			%imageData = read(t);

			
		    
		    Image = double(Image);
		    %image = imrotate(Image, 90);
			%Image = flipdim(image, 2);
		    %fullfile(filename,dicom_file);
			%rSlope = info.RescaleSlope;
			%rinter = info.RescaleIntercept;
			hounsfieldImage = Get_hounsfieldImage(Image);
			%%%%%%%%%%% Generating Mask %%%%%%%%%%%%%%%%%
			
			curr_mask = squeeze(double(Mask(slice_index, : , :)));
			se = strel('disk', 4);
			%curr_mask = imrotate(curr_mask, 180);
			curr_mask = imfill(curr_mask, 'holes');
			seg_Image =  hounsfieldImage .* curr_mask;
			org_seg_Image = Image .* curr_mask;


			
			%onethird = ceil((len/3));
			%twothird = len - onethird;

			%start_ = 7;
			%end_ =numel(dicomfiles) - 7;
			flag =1;
			mid = floor(len/2);
			if( len > 80)
	    		start_ = mid -40 ;
	    		end_ = mid + 40;
			
			else
	    		start_ = mid - 25;
	    		end_ = mid + 25;
	    	end
	    	
	    	%start_ = 10;
	    	%end_ = len-1;
			if ((~ all(seg_Image(:) < 2) & flag == 1) & ~(slice_index < start_) & ~(slice_index > end_))%(numel(dicomfiles) - 5)
				%dicom_file = niftifiles(slice_index).name;
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
				catch
    				warning('Ill-conditioned covariance created in processing');
    
				end
				
				%{
				gmm_x = linspace(min_val,max_val,num_bins);
				gmm_x = reshape(gmm_x, [], 1);
				gmm_y = pdf(GMModel, gmm_x)%@(gmm_x) arrayfun(@(x0) pdf(GMModel,[x0]),x);
				%}
				img_reco = reshape(seg_Image,[],1);
				clusterX = cluster(GMModel,img_reco);

				%{
				f = figure
				hist = histogram(S, num_bins, 'Normalization','pdf');
				hold on
				plot(gmm_x, gmm_y,"Color",'r')
				%}
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
				org_seg_Image = final_mask_1 .* org_seg_Image;

				%x = imgHyperb2D(org_seg_Image);


				curr_mask_prewitt = edge(curr_mask,'Prewitt');
				curr_mask_roberts = edge(curr_mask,'roberts');
				curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2) * 950);

				my_img= org_seg_Image + curr_mask_final;
				x = imgHyperb2D(my_img);
				volume = my_img;
				min_ = min(min(volume));
				max_ = max(max(volume));
				volume(volume < min_) = min_;
				volume(volume > max_) = max_;
				volume = (volume - min_) / (max_ - min_);
				volume = uint8(volume.*255);

				%{
				if strcmp(subFolder, 'Train') 
		    		dest_dir = fullfile(outputDir,  patients_name);
		    	else
		    		dest_dir = fullfile(outputDir, patients_name);
		    	end
		    	%}
		    	dest_dir = fullfile(outputDir, patients_name);
		    	if ~exist(dest_dir, 'dir')
		    		mkdir(dest_dir)
		        end
				fname = [dest_dir, '/', patients_name, '_', num2str(slice_index),'.jpg'];
				%parsave(fname, final_processed_img)
				parsave(fname, volume)
			end
			
		end
	
	%Severity_score_patient{pat_index, 1} = patients_name;
	%Severity_score_patient{pat_index, 1} = severity_scores;
	end
		
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

function parsave(fname, x)
	%f = figure;
	%imshow(mat2gray(x));
	%saveas(x,fname,'png'); 
	%close(f);
  	imwrite(x, fname);
  	%save(fname,'x');
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

function [image] = Generate_mask_image(img, final_img)
    
    image = final_img;
    for j = 1 : size(img, 1) % This loop multiply each voxel value by the rescale slope
        for i = 1 : size(img, 2)
            if (img(i,j) == 750)
                image(i,j) = img(i,j);
            end
            
        end
    end
end


function [image] = Generate_final_image(img, final_img)
    
    image = img;
    for j = 1 : size(img, 1) % This loop multiply each voxel value by the rescale slope
        for i = 1 : size(img, 2)
            if (img(i,j) ==0 )
                image(i,j) = final_img(i,j);
            end
            
        end
    end
end


function [severity_score] = find_severity(final_processed_img, curr_mask)

    lung_area = nnz(curr_mask);
    infected_area = nnz(final_processed_img);
    severity_score = infected_area / lung_area;
    
end

function outV = imgHyperb2D(img1)
	%clear outV;
	volRS=uint16(img1);
	c=0.04;
	MAX=65535;% 255 to 8 bits, 65535 to 16 bits
	[xx yy]=size(volRS(:,:));
	[a b]=histc(reshape(volRS(:,:),1,xx*yy),[1:max(max(volRS(:,:)))]);% a is the histogram, b the values
	% normalize it
	norma=a/max(a);
	% calculates the cumulative histogram and normalize it
	cuma=cumsum(norma);
	normcm=cuma/max(cuma);
	% aplies formula to all pixels
	%[xx yy]=size(volRS(:,:,100));
	for i=1:xx
		for j=1:yy
	    	% 16 bits
	    	if MAX==65535
	        	if(volRS(i,j)==0)
	            	outV(i,j)=uint16(MAX*c*((exp(log(1+(1/c))*normcm(volRS(i,j)+1)-1))));
	        	else
	            	outV(i,j)=uint16(MAX*c*((exp(log(1+(1/c))*normcm(volRS(i,j))-1))));    
	        	end
	    	% 8 bits
	    	else
	        	if MAX==255
	            	if(volRS(i,j)==0)
	                	outV(i,j)=uint16(MAX*c*((exp(log(1+(1/c))*normcm(volRS(i,j)+1)-1))));
	            	else
	                	outV(i,j)=uint16(MAX*c*((exp(log(1+(1/c))*normcm(volRS(i,j))-1))));    
	            	end
	        	end
	    	end
		end
	end
end