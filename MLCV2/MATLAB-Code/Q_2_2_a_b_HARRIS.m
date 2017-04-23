%% Q2_2_a_b HARRIS
% a) Estimate fundamental matrix using list of correspondences from Q1.1 or 
% Q1.2.a.
% b) Calculate the epipoles for images A and B. Show epipolar lines and 
% epipoles on the images if possible.

%% parameters to be adjusted
threshold = 0.01;  % 1 percent of maximum response value
visualise = 1;
knn_threshold = 1.11;
num_strongest = 500; % number of strongest interest points
ANMS = true; % adaptive non-maximal suppression
radius = 5;
descriptor_type = 'CIRCLE';
descriptor_diameter = 32; % diameter of circle descriptor
num_manual_IPs = 10; % number of interest points get from manual selection

%% read image
img1 = imread('pictures/FD1.JPG');
img1 = imresize(img1,[680 850]);
img2 = imread('pictures/FD2.JPG');
img2 = imresize(img2,[680 850]);

if (descriptor_type ~= 'CIRCLE')
    % initialisation
    addpath('external/vlfeat-0.9.18');
    addpath('external/libsvm-3.18/matlab');
    run('external/vlfeat-0.9.18/toolbox/vl_setup.m'); % vlfeat library
    cd('external/libsvm-3.18/matlab'); % libsvm library
    run('make');
    cd('../../..');
end

% ================= convert colour scale to gray scale =================
% img 1
if (size(size(img1),2) == 3)
    img1 = rgb2gray(img1);
end

% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end


%% extract interest points & their descriptors
% image 1 Harris points and feature descriptors
% =========================================================================
points1_raw = InterestPointDetector(img1, threshold, ANMS, radius);
if size(points1_raw,1)>num_strongest
    points1 = points1_raw(1:num_strongest,:);
else
    points1 = points1_raw;
end

if (descriptor_type == 'CIRCLE')
    % Obtain circle patches as descriptors
    descriptor1 = getDescriptor(points1(:,1:2), img1, descriptor_diameter);
elseif (descriptor_type == 'SIFT')
    % Obtain SIFT descriptor
    descriptor1 = [];
    for i=1:size(points1,1)
        img1_    = vl_imsmooth(im2double(img1), sqrt(points1(i,3)^2 - 0.5^2));
        [Ix, Iy] = vl_grad(img1_) ;
        mod      = sqrt(Ix.^2 + Iy.^2) ;
        ang      = atan2(Iy,Ix) ;
        grd      = shiftdim(cat(3,mod,ang),2) ;
        grd      = single(grd) ;
        descriptor1(i,:) = vl_siftdescriptor(grd, ...
        [points1(i,2),points1(i,1),points1(i,3:4)]') ;
    end
end

% image 2 Harris points and feature descriptors
% =========================================================================
points2_raw = InterestPointDetector(img2, threshold, ANMS, radius);
if size(points2_raw,1)>num_strongest
    points2 = points2_raw(1:num_strongest,:);
else
    points2 = points2_raw;
end

if (descriptor_type == 'CIRCLE')
    % Obtain circle patches as descriptors
    descriptor2 = factor*getDescriptor(points2(:,1:2), img2, descriptor_diameter/factor);
elseif (descriptor_type == 'SIFT')
    % Obtain SIFT descriptor
    descriptor2 = [];
    for i=1:size(points2,1)
        img2_    = vl_imsmooth(im2double(img2), sqrt(points2(i,3)^2 - 0.5^2));
        [Ix, Iy] = vl_grad(img1_) ;
        mod      = sqrt(Ix.^2 + Iy.^2) ;
        ang      = atan2(Iy,Ix) ;
        grd      = shiftdim(cat(3,mod,ang),2) ;
        grd      = single(grd) ;
        descriptor2(i,:) = vl_siftdescriptor(grd, ...
        [points2(i,2),points2(i,1),points2(i,3:4)]') ;
    end
end

%% Visualise Detected Interest Points
if visualise
    f1=figure;
    subplot(1,2,1);
    myvisualise(points1,img1,'Img1 - interest points');
    subplot(1,2,2);
    myvisualise(points2,img2,'Img2 - interest points')
    set(f1, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     print(f1,'-dpng','-r600');
end

%% Matching correspondance 
[matchedPoints1, matchedPoints2] = myFeatureMatching(points1(:,1:2),...
                points2(:,1:2), descriptor1, descriptor2, knn_threshold);

%% Visualise matched correspondences
if visualise
    figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1),fliplr(matchedPoints2),'montage','Parent',ax);
    title(ax, 'Matched Correspondances before RANSAC');
    legend(ax, 'Matched points 1','Matched points 2');
end

%% Compute Homography Matrix using RANSAC
numberoftrial = 10000;
threshold = 20; % average cityblock distance in pixels

[besth, f2, inliersIndex] = RansacPredictH(matchedPoints1, matchedPoints2, numberoftrial, threshold);
matchedPoints1_inliers_ground_truth = [];
matchedPoints2_inliers_ground_truth = [];
cnt = 1;
for i=1:size(matchedPoints1,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers_ground_truth(cnt,:) = matchedPoints1(i,:);
        matchedPoints2_inliers_ground_truth(cnt,:) = matchedPoints2(i,:);
        cnt = cnt +1;
    end
end
projectedv2 = hmatrixproject(matchedPoints1_inliers_ground_truth, besth);

%% visualise projected points after RANSAC
if visualise
    figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers_ground_truth),fliplr(matchedPoints2_inliers_ground_truth),'montage','Parent',ax);
    title(ax, 'Inliers after RANSAC');
    legend(ax, 'Matched points 1','Matched points 2');
end

%% Compute Fundamental Matrix using RANSAC
numOfTrials = 10000;
thr = 0.01;
[bestf, f, inliers] = RansacPredictF(matchedPoints1_inliers_ground_truth,...
                      matchedPoints2_inliers_ground_truth,numOfTrials,thr);
matchedPoints1_inliers = matchedPoints1_inliers_ground_truth.*[inliers',inliers'];
matchedPoints2_inliers = matchedPoints2_inliers_ground_truth.*[inliers',inliers'];
% f1 = estimateFundamentalMatrix(v1, v2,'Method','RANSAC','NumTrials',500);
figure;
drawEpipolarline(bestf, matchedPoints1_inliers, img2);

%% visualise projected points after RANSAC
if visualise
    figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers),fliplr(matchedPoints2_inliers),'montage','Parent',ax);
    title(ax, 'Inliers after Fundamental Matrix RANSAC');
    legend(ax, 'Matched points 1','Matched points 2');
end