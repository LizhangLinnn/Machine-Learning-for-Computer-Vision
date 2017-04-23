img1 = imread('pictures/img1.pgm');
img1 = imresize(img1,[680 850]);
img2 = imread('pictures/img5.pgm');
img2 = imresize(img2,[680 850]);
% img1 = imread('pictures/HG1.JPG');
% img1 = imresize(img1,[680 850]);
% img2 = imread('pictures/HG1.JPG');
% img2 = imresize(img2,[680/2 850/2]);

% ================= convert colour scale to gray scale =================
% img 1
if (size(size(img1),2) == 3)
    img1 = rgb2gray(img1);
end

% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end


%% paramaters that can be adjusted
threshold = 0.01;  % 1 percent of maximum response value
visualise = 1;
knn_threshold = 1.2;
num_strongest = 500; % number of strongest interest points
ANMS = true; % adaptive non-maximal suppression
radius = 2;

%% image 1
% image 1 - Harris Point Detector
% coords1 = InterestPointDetector(img1, threshold, ANMS, radius);
points1_raw = InterestPointDetector_Harris_Laplace(img1,threshold);
if size(points1_raw,1)>num_strongest
    points1 = points1_raw(1:num_strongest,:);
else
    points1 = points1_raw;
end
coords1 = points1(:,1:2);

% Obtain Descriptor
diameter = 32;
descriptor1 = [];
for i=1:size(points1,1)
scale = points1(i,3)/1.5;
descriptor1(i,:) = getDescriptor(coords1(i,:), img1, diameter*scale)/scale^2;
end
%% image1 get SIFT descriptor
% descriptor1 = [];
% for i=1:size(points1,1)
% img1_    = vl_imsmooth(im2double(img1), sqrt(points1(i,3)^2 - 0.5^2));
% [Ix, Iy] = vl_grad(img1_) ;
% mod      = sqrt(Ix.^2 + Iy.^2) ;
% ang      = atan2(Iy,Ix) ;
% grd      = shiftdim(cat(3,mod,ang),2) ;
% grd      = single(grd) ;
% descriptor1(i,:) = vl_siftdescriptor(grd, ...
%     [points1(i,2),points1(i,1),points1(i,3:4)]') ;
% end


%% image 2
%Q1 2a) image 2 - Harris Point Detector
% coords2 = InterestPointDetector(img2, threshold, ANMS, radius);
points2_raw = InterestPointDetector_Harris_Laplace(img2,threshold);
if size(points2_raw,1)>num_strongest
    points2 = points2_raw(1:num_strongest,:);
else
    points2 = points2_raw;
end
coords2 = points2(:,1:2);

% Obtain Descriptor
diameter = 32;
descriptor2 = [];
for i=1:size(points2,1)
scale = points2(i,3)/1.5;
descriptor2(i,:) = getDescriptor(coords2(i,:), img2, diameter*scale)/scale^2;
end
%% image2 get SIFT descriptor
% descriptor2 = [];
% for i=1:size(points2,1)
% img2_    = vl_imsmooth(im2double(img2), sqrt(points2(i,3)^2 - 0.5^2));
% [Ix, Iy] = vl_grad(img2_) ;
% mod      = sqrt(Ix.^2 + Iy.^2) ;
% ang      = atan2(Iy,Ix) ;
% grd      = shiftdim(cat(3,mod,ang),2) ;
% grd      = single(grd) ;
% descriptor2(i,:) = vl_siftdescriptor(grd,...
%     [points2(i,2),points2(i,1),points2(i,3:4)]') ;
% end

%% visualisation for ANMS
% if visualise
%     f1=figure;
%     subplot(1,2,1);
%     myvisualise(coords1_no_ANMS(1:500,:), img1, 'Strongest 500 IPs (without suppresion)');
%     subplot(1,2,2);
%     myvisualise(coords1_r_10(1:500,:), img1, 'Strongest 500 IPs (with suppresion) and radius=10');
%     set(f1, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     print(f1,'-dpng','-r600');
% end

%% visualisation for Interest Points ** before matching **
if visualise
    f1=figure;
    subplot(1,2,1);
    myvisualise(points1,img1,sprintf('Detected Features, r=%d',radius));
    subplot(1,2,2);
    myvisualise(points2,img2,sprintf('Detected Features, r=%d',radius))
    set(f1, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     print(f1,'-dpng','-r600');
end

%% Matching correspondance (our own implementation)
% ========== Q1 2C) ========== 
[idx_matchedIP_img2_raw, distance] = myKnnsearch(descriptor1,descriptor2,knn_threshold);
[distance,idx] = sort(distance,'ascend');
matchedIP_idx = idx(~isinf(distance)); % the index of matched interest points

idx_matchedIP_img1_raw = 1:1:size(idx_matchedIP_img2_raw,2);
idx_matchedIP_img1 = idx_matchedIP_img1_raw(matchedIP_idx);
idx_matchedIP_img2 = idx_matchedIP_img2_raw(matchedIP_idx);

% remove correspondance that are not unique (which are ambiguous)
if knn_threshold ~= 1
    arr1 = getNonRepeatableElementIdx(idx_matchedIP_img1);
    arr2 = getNonRepeatableElementIdx(idx_matchedIP_img2);
    idx = intersect(arr1,arr2);
    correspondance = [idx_matchedIP_img1(idx);idx_matchedIP_img2(idx)];
    if size(correspondance,2)==0
       error('user defined error - no interest point is matched under current parameters'); 
    end
else
    correspondance = [idx_matchedIP_img1;idx_matchedIP_img2];
end

matchedPoints1=points1(correspondance(1,:),1:2);
matchedPoints2=points2(correspondance(2,:),1:2);

% visualise matched correspondences
if visualise
figure; ax = axes;
showMatchedFeatures(img1,img2,fliplr(matchedPoints1),fliplr(matchedPoints2),'montage','Parent',ax);
title(ax, 'our own implementation');
legend(ax, 'Matched points 1','Matched points 2');
end

%% Compute homography matrix
% ========== Q1 3A) ==========  

numberoftrial = 10000;
threshold = 10; % 0.001 pixels
[besth, f2, inliersIndex] = RansacPredictH(matchedPoints1, matchedPoints2, numberoftrial, threshold);

if visualise
    figure;
    subplot(2,2,1);
    mytitle = 'Image 1 - Correspondance before RANSAC';
    myvisualise(matchedPoints1, img1, mytitle);
    subplot(2,2,2);
    mytitle = 'Image 2 - Correspondance before RANSAC';
    myvisualise(matchedPoints2, img2, mytitle);
end
matchedPoints1_inliers = [];
matchedPoints2_inliers = [];
cnt = 1;
for i=1:size(matchedPoints1,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers(cnt,:) = matchedPoints1(i,:);
        matchedPoints2_inliers(cnt,:) = matchedPoints2(i,:);
        cnt = cnt +1;
    end
end

projectedv2 = hmatrixproject(matchedPoints1_inliers, besth);

if visualise
    subplot(2,2,[3 4]);
    myvisualise(projectedv2, img2, 'Projected coords1 on image 2');
    title('projected inliers');
%     % super title displaying the values of used parameters 
%     if ANMS 
%         print_radius = sprintf(' suppresion radius = %d pixel',radius);
%     end
%     print_thre = sprintf('\n threshold for local maxima = %f',threshold);
%     print_knnThre = sprintf('\n knn threshold for rejecting ambiguous matchings = %f',knn_threshold);
%     suptitle_print_msg = strcat(print_radius,print_thre,print_knnThre);
%     suptitle(suptitle_print_msg);
end

% Compute HA error
HA = calculateHA(projectedv2, matchedPoints2_inliers);

%%
% visualise matched correspondences after RANSAC
if visualise
figure; ax = axes;
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers),fliplr(matchedPoints2_inliers),'montage','Parent',ax);
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift(inliersIndex,:)),fliplr(matchedPoints2_sift(inliersIndex,:)),'Parent',ax);
title(ax, 'Matched Correspondences after RANSAC');
legend(ax, 'Matched points 1','Matched points 2');
end