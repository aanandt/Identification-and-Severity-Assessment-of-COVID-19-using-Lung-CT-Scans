function[Infection_mask] = InfectionMask_Generation(Imagepath, Maskpath, index)
    
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
    final_mask = bwareaopen(final_mask, 70);
    se = strel('disk', 2);
    final_mask = imfill(final_mask, 'holes');

    segmented_img = final_mask .* seg_Image;
    
    
    
    Infection_mask = segmented_img;
    Infection_mask(Infection_mask == -1024) = 0;
    Infection_mask(Infection_mask ~= 0) = 1;        
    se = strel('disk', 3);
    Infection_mask = imdilate(Infection_mask,se);
end