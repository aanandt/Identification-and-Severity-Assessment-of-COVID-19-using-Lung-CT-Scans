function [] = preprocessing()

	inputDir = '../../../Dataset/'
	outputDir = 'Preprocessed_Data_AbelationStudy_Updated/SPGC_Train_New/'
	%input_file = '../Dataset/Slice_level_labels.csv';
	Mask_path = '../../../TempDir/Matlab_Mask_files_Train/';
	input_file = '../../../Dataset/Slice_level_labels_dummy.csv';
	
	slice_level_labels = Get_slice_level_label(input_file);
	Severity_score_patient = {};
	subFolders = Read_folders(inputDir);
	
	for i = 2 : 1 : 2%numel(subFolders)
		subFolder = subFolders(i).name;
		folder_path = fullfile(inputDir,subFolder);
		categories = Read_folders(folder_path);
		for j = 1 : 1 : numel(categories)
			category = categories(j).name;
			folder_path = fullfile(inputDir,subFolder,category);
			subsets = Read_folders(folder_path);

			if (strcmp( subFolder, 'Train'))
				if strcmp(category, 'Normal')
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, 'Negative', outputDir, Mask_path, slice_level_labels, Severity_score_patient);
				
				else
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, 'Positive', outputDir, Mask_path, slice_level_labels, Severity_score_patient);
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, 'Positive', outputDir, Mask_path, slice_level_labels, Severity_score_patient);
				end
			end
			%{
			else
				if strcmp(category, 'Normal')
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, 'Negative', outputDir, Mask_path, slice_level_labels, Severity_score_patient);
				
				else
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, subsets(1).name, outputDir, Mask_path, slice_level_labels, Severity_score_patient);
					Extract_Dicom_Preprocessing(folder_path, category, subFolder, subsets(2).name, outputDir, Mask_path, slice_level_labels, Severity_score_patient);
				end

			end
			%}	

		end
		Severity_score_patient = [Severity_score_patient_1; Severity_score_patient_2; Severity_score_patient_3; Severity_score_patient_4, Severity_score_patient_5, Severity_score_patient_6];
		filename = 'Severity_score_patient_wise';
		cell2csv(filename,Severity_score_patient)
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

function [Severity_score_patient] = Extract_Dicom_Preprocessing(folder_path, category, subFolder, subset, outputDir, Mask_path, slice_level_labels, Severity_score_patient)
	
	folder_name = fullfile(folder_path, subset);
	patients = Read_folders(fullfile(folder_path, subset));
	delete(gcp('nocreate'));
	parpool(16)
	
	%Slcie_level_infection = {};
	parfor pat_index = 1 : 1 : numel(patients)

		patients_name = patients(pat_index).name;
		filename = fullfile(folder_name, patients_name);
		dicomfiles = dir(fullfile(filename, '*.dcm'))
		mask_filename = fullfile(Mask_path, patients_name, 'mask.mat');
		Mask = load(mask_filename).data();
		%severity_scores = [];
		for slice_index = 1 : 1: numel(dicomfiles)


			dicom_file = dicomfiles(slice_index).name;
			Image = dicomread(fullfile(filename,dicom_file));
		    info = dicominfo(fullfile(filename,dicom_file));
		    
		    %patients_name
		    %{
		    if (~strcmp(patients_name, 'cap007') & info.InstanceNumber >= 113)
				continue;
			end
			if (~strcmp(patients_name ,'cap025') & info.InstanceNumber >= 105)
				continue;
			end
			if (~strcmp(patients_name, 'cap060') & info.InstanceNumber >= 59)
				continue;
			end
			%}
		    Image = double(Image);
			rSlope = 1; %info.RescaleSlope;
			rinter = -1024; %info.RescaleIntercept;
			hounsfieldImage = Get_hounsfieldImage(Image, info);
		
			%%%%%%%%%%% Gnerating Mask %%%%%%%%%%%%%%%%%
			curr_mask = squeeze(double(Mask(info.InstanceNumber, : , :)));
			seg_Image = hounsfieldImage .* curr_mask;
			org_seg_Image = Image .* curr_mask;

			flag =1;
			temp = split(dicom_file, '_');
		    temp1 = split(temp{2}, '.');
		    dicom_index = str2num(temp1{1});
		    
			if (strcmp(subFolder,'Train') & ~strcmp(category,'Normal'))
				label_index = find(contains(slice_level_labels(:,1),patients_name))
				%disp('hi')
				%flag = slice_level_labels{label_index,2}(info.InstanceNumber)%(dicom_index)
				
				if ~isempty(label_index)
					flag = slice_level_labels{label_index,2}(info.InstanceNumber);%(dicom_index)
					class(label_index)
				else
					
					flag = 0;
		    	end
		    	
		    end

			mid = floor((numel(dicomfiles)/2));
		   
		    
    		if( numel(dicomfiles) > 80)
        		start_ = mid -40 ;
        		end_ = mid + 40;
    		
    		else
        		start_ = mid - 20;
        		end_ = mid + 20;
        	end
    		
    		
        	disp('+++++++++++++++++++++++++++++++++++++++++')
		    disp("InstanceNumber " + string(info.InstanceNumber))
		    disp(string(start_) + string(end_))
		    disp("flag "+ num2str(flag))
		    disp('-----------------------------------------')
			if ((~ all(seg_Image(:) < 2) & flag == 1) & ~(info.InstanceNumber < start_) & ~(info.InstanceNumber > end_))%((numel(dicomfiles)- 25))
				
				x = ['Preprocessing strated for patient:', patients_name,  '\tslice number:', num2str(info.InstanceNumber)];
				disp(x)
				%%%%%%%%%% Fit GMM on Histogram of Intensities%%%%%%%%%%%%%%%%%
				S = reshape(seg_Image,[],1);
				max_val = max(max(seg_Image));
				min_val = min(min(seg_Image));
				num_bins = ceil((max_val - min_val)/3);
				S = S(S ~= 0);
				GMModel = fitgmdist(S,3);
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
				GMM_wo_fill_morpho_seg = final_mask .* seg_Image;
				GMM_wo_fill_morpho_org_seg = final_mask .* org_seg_Image;

				%seg_Image = final_mask_1 .* seg_Image;
				%org_seg_Image = final_mask_1 .* org_seg_Image;

				%x = imgHyperb2D(org_seg_Image);

				SaveImage(GMM_wo_fill_morpho_org_seg, outputDir, 'GMM_filter_img_wo_imfill', category, patients_name, slice_index)



				GMM_fill_morpho_seg = final_mask_1 .* seg_Image;
				GMM_fill_morpho_org_seg = final_mask_1 .* org_seg_Image;

				%seg_Image = final_mask_1 .* seg_Image;
				%org_seg_Image = final_mask_1 .* org_seg_Image;

				%x = imgHyperb2D(org_seg_Image);

				SaveImage(GMM_fill_morpho_org_seg, outputDir, 'GMM_filter_img_wo_boundary', category, patients_name, slice_index)
				curr_mask_prewitt = edge(curr_mask,'Prewitt');
				curr_mask_roberts = edge(curr_mask,'roberts');
				curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2) * 950);

				my_img= GMM_fill_morpho_org_seg + curr_mask_final;

				SaveImage(my_img, outputDir, 'GMM_filter_img_with_imfillboundary', category, patients_name, slice_index)

				GMM_filter_img = (final_mask .* org_seg_Image);

			    curr_mask_prewitt = edge(curr_mask,'Prewitt');
				curr_mask_roberts = edge(curr_mask,'roberts');
				curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2) * 950);

				my_img= GMM_filter_img + curr_mask_final;
				SaveImage(my_img, outputDir,  'GMM_filter_img_woimfill_withboundary', category, patients_name, slice_index)

    			
			    V1 = vesselness2D((seg_Image), 0.5:0.5:2.5, [1;1], 1, true);
			    V1(V1 < 0.75) = 0;
			    V1(V1 ~= 0) = 1;
			    final_mask = final_mask_1 .* ~V1;
			    final_mask = bwareaopen(final_mask, 70);
			    se = strel('disk', 4);
			    final_mask = imfill(final_mask, 'holes');

			    %segmented_img = final_mask .* seg_Image;
			    segmented_img = final_mask .* org_seg_Image;
			    
			    V2 = vesselness2D((seg_Image), 0.5:0.5:2.5, [1;1], 1, true);
			    V2(V2 < 0.6) = 0;
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
			    
			    %seg_Image(seg_Image ==0)=-1024;
    			%segmented_img(segmented_img ==0)=-1024;

			    curr_mask_prewitt = edge(curr_mask,'Prewitt');
			    curr_mask_roberts = edge(curr_mask,'roberts');
			    curr_mask_final = (ceil((curr_mask_prewitt + curr_mask_roberts)/2) * 950);
			    final_processed_img = segmented_img + curr_mask_final;
			    %{
			    for j = 1 : size(curr_mask_final, 1) % This loop multiply each voxel value by the rescale slope
			        for i = 1 : size(curr_mask_final, 2)
			            if (curr_mask_final(i,j) == 250)
			                segmented_img(i,j) = curr_mask_final(i,j);
			            end
			            
			        end
			    end
			    %}
			    			    
			    %%%%%%%%%%%Windowing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			    med_window_img = Windowing(seg_Image, 350, 50);
			    med_window = med_window_img;
			    
			    med_window(med_window ~= -125) = 1;
			    med_window(med_window == -125) = 0;
			    med_window_mask = bwareaopen(med_window, 70);
			    %img= med_window .* (org_seg_Image);
			    final_mask = ceil((final_mask + med_window_mask) / 2);
			    %final_mask = ceil((final_mask + curr_mask_final) / 2);
			    
			    
			    my_img= final_mask .* org_seg_Image;
			    %my_img(my_img <= -1024) = 0;
			    SaveImage(my_img, outputDir, 'my_img',  category, patients_name, slice_index);

				SaveImage(final_processed_img, outputDir, 'Final_preprocessed_img', category, patients_name, slice_index);
				
				SaveImage(segmented_img, outputDir, 'Final_preprocessed_img_without_border', category, patients_name, slice_index);
				

				

			    Mera_mask = my_img;
			    Mera_mask(Mera_mask <= -1024) = 0;
			    Mera_mask(Mera_mask ~= 0) = 1;      

				%{
				if strcmp(subFolder, 'Train') 
		    		dest_dir = fullfile(outputDir,  patients_name);
		    	else
		    		dest_dir = fullfile(outputDir, patients_name);
		    	end
		    	%}
		    	%{

				my_img= org_seg_Image + curr_mask_final;
				x = my_img;%imgHyperb2D(my_img);
				volume = x;
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
				fname = [dest_dir, '/', patients_name, '_', num2str(info.InstanceNumber),'.png'];
				%parsave(fname, final_processed_img)
				parsave(fname, volume)
				%}
			end
		end
		%Severity_score_patient{pat_index, 1} = patients_name;
		%Severity_score_patient{pat_index, 1} = severity_scores;
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

function [] = SaveImage(my_img, outputDir, category, sub_category, patients_name, slice_index)

	volume = my_img;
    %min_ = -1150;
    %max_ = 150;
    min_ = min(min(volume));
    max_ = max(max(volume));
    volume(volume < min_) = min_;
    volume(volume > max_) = max_;
    volume = (volume - min_) / (max_ - min_);
    volume = uint8(volume.*255);
    dest_dir = fullfile(outputDir, category, sub_category, patients_name);
	if ~exist(dest_dir, 'dir')
		mkdir(dest_dir)
    end
	fname = [dest_dir, '/', patients_name, '_', num2str(slice_index),'.jpg'];
	parsave(fname, volume)



end

function parsave(fname, x)
	%f = figure;
	%imshow(x);
	%saveas(f,fname); 
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
