% SIFT
%% parameters to be adjusted
knn_threshold = 1.3;
num_strongest = 500; % number of strongest interest points
visualise = 1;
%% read image
% img1 = imread('pictures/scene1.ppm');
% img2 = imread('pictures/scene2.ppm');
% img1 = imread('pictures/img1.pgm');
% img2 = imread('pictures/img6.pgm');
img1 = imread('pictures/HG1.JPG');
img1 = imresize(img1,[680 850]);
img2 = imread('pictures/HG2.JPG');
img2 = imresize(img2,[680 850]);


addpath('external/vlfeat-0.9.18');
addpath('external/libsvm-3.18/matlab');
run('external/vlfeat-0.9.18/toolbox/vl_setup.m'); % vlfeat library
cd('external/libsvm-3.18/matlab'); % libsvm library
run('make');
cd('../../..');

% ================= convert colour scale to gray scale =================
% img 1
if (size(size(img1),2) == 3)
    img1 = rgb2gray(img1);
end
img1 = im2double(img1);
% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end
img2 = im2double(img2);


%% extract interest points & their descriptors
[F1_raw,D1_raw]=vl_sift(single(img1));
[F2_raw,D2_raw]=vl_sift(single(img2));
coords1_sift_raw = horzcat(F1_raw(2,:)',F1_raw(1,:)');
coords2_sift_raw = horzcat(F2_raw(2,:)',F2_raw(1,:)');

%% get strong interest points
[~,idx1] = sort(F1_raw(4,:),'descend');
[~,idx2] = sort(F2_raw(4,:),'descend');

%% get the strongest interest points and their descriptors
F1 = F1_raw(:,idx1(1:num_strongest));
F2 = F2_raw(:,idx2(1:num_strongest));
D1 = D1_raw(:,idx1(1:num_strongest));
D2 = D2_raw(:,idx2(1:num_strongest));
coords1_sift = coords1_sift_raw(idx1(1:num_strongest),:);
coords2_sift = coords2_sift_raw(idx2(1:num_strongest),:);
points1_sift = horzcat(coords1_sift,F1(3:4,:)');
points2_sift = horzcat(coords2_sift,F2(3:4,:)');


%% ************ extract descriptor using vl_siftdescriptor ************ 
%     % img1
%     D1 = [];
%     for i=1:size(F1,2)
%     img1_    = vl_imsmooth(im2double(img1), sqrt(F1(3,i)^2 - 0.5^2));
%     [Ix, Iy] = vl_grad(img1_) ;
%     mod      = sqrt(Ix.^2 + Iy.^2) ;
%     ang      = atan2(Iy,Ix) ;
%     grd      = shiftdim(cat(3,mod,ang),2) ;
%     grd      = single(grd) ;
%     D1(:,i) = vl_siftdescriptor(grd, F1(:,i)) ;
%     end
%     % img2
%     D2 = [];
%     for i=1:size(F2,2)
%     img2_    = vl_imsmooth(im2double(img2), sqrt(F2(3,i)^2 - 0.5^2));
%     [Ix, Iy] = vl_grad(img2_) ;
%     mod      = sqrt(Ix.^2 + Iy.^2) ;
%     ang      = atan2(Iy,Ix) ;
%     grd      = shiftdim(cat(3,mod,ang),2) ;
%     grd      = single(grd) ;
%     D2(:,i) = vl_siftdescriptor(grd, F2(:,i)) ;
%     end

%% visualise the strongest N points in both images
if visualise
    figure;
    subplot(2,2,1);
%     myvisualise(coords1_sift, img1, 'very original corner image 1');
    myvisualise(points1_sift, img1, 'very original corner image 1');
    subplot(2,2,2);
    myvisualise(points2_sift, img2, 'very original corner image 2');
end

%% Matching correspondence (from external library)
THRESH = 1.8;
matches_sift = vl_ubcmatch(D1,D2,THRESH);
matchedPoints1_sift=coords1_sift(matches_sift(1,:),:);
matchedPoints2_sift=coords2_sift(matches_sift(2,:),:);
% visualise matched correspondences
if visualise
figure; ax = axes;
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
title(ax, 'external library');
legend(ax, 'Matched points 1','Matched points 2');
end

%% Matching correspondence (our own implementation)
[idx_matchedIP_img2_sift_raw, distance] = myKnnsearch(D1',D2',knn_threshold);
[distance,idx] = sort(distance,'ascend');
matchedIP_sift_idx = idx(~isinf(distance)); % the index of matched interest points

idx_matchedIP_img1_sift_raw = 1:1:size(idx_matchedIP_img2_sift_raw,2);
idx_matchedIP_img1_sift = idx_matchedIP_img1_sift_raw(matchedIP_sift_idx);
idx_matchedIP_img2_sift = idx_matchedIP_img2_sift_raw(matchedIP_sift_idx);

% remove correspondance that are not unique (which are ambiguous)
if knn_threshold ~= 1
    arr1 = getNonRepeatableElementIdx(idx_matchedIP_img1_sift);
    arr2 = getNonRepeatableElementIdx(idx_matchedIP_img2_sift);
    idx = intersect(arr1,arr2);
    correspondance_sift = [idx_matchedIP_img1_sift(idx);idx_matchedIP_img2_sift(idx)];
    if size(correspondance_sift,2)==0
       %error('user defined error - no interest point is matched under current parameters'); 
    end
else
    correspondance_sift = [idx_matchedIP_img1_sift;idx_matchedIP_img2_sift];
end
matchedPoints1_sift=coords1_sift(correspondance_sift(1,:),:);
matchedPoints2_sift=coords2_sift(correspondance_sift(2,:),:);

%%
% visualise matched correspondences
if visualise
figure; ax = axes;
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift(inliersIndex,:)),fliplr(matchedPoints2_sift(inliersIndex,:)),'Parent',ax);
title(ax, 'our own implementation');
legend(ax, 'Matched points 1','Matched points 2');
end

%% Compute homography matrix
% ========== Q1 3A) ==========  

numberoftrial = 100;
threshold = 0.5; % pixels
[besth, f2, inliersIndex] = RansacPredictH(matchedPoints1_sift, matchedPoints2_sift, numberoftrial, threshold);

if visualise
    figure;
    subplot(2,2,1);
    mytitle = 'Image 1 - Correspondance before RANSAC';
    myvisualise(matchedPoints1_sift, img1, mytitle);
    subplot(2,2,2);
    mytitle = 'Image 2 - Correspondance before RANSAC';
    myvisualise(matchedPoints2_sift, img2, mytitle);
end
matchedPoints1_inliers_sift = [];
matchedPoints2_inliers_sift = [];
cnt = 1;
for i=1:size(matchedPoints1_sift,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers_sift(cnt,:) = matchedPoints1_sift(i,:);
        matchedPoints2_inliers_sift(cnt,:) = matchedPoints2_sift(i,:);
        cnt = cnt +1;
    end
end

projectedv2 = hmatrixproject(matchedPoints1_inliers_sift, besth);

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
HA = calculateHA(projectedv2, matchedPoints2_inliers_sift);

%%
% visualise matched correspondences after RANSAC
if visualise
figure; ax = axes;
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers_sift),fliplr(matchedPoints2_inliers_sift),'montage','Parent',ax);
% showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift(inliersIndex,:)),fliplr(matchedPoints2_sift(inliersIndex,:)),'Parent',ax);
title(ax, 'Matched Correspondences after RANSAC');
legend(ax, 'Matched points 1','Matched points 2');
end