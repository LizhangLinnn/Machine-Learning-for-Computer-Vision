%% Q2_2_a SIFT
% Estimate fundamental matrix using list of correspondences from Q1.1 or 
% Q1.2.a.

%% parameters to be adjusted
knn_threshold = 1.4;
num_strongest = 500; % number of strongest interest points
visualise = 1;

%% read image
img1 = imread('pictures/FD7.JPG');
% img1 = imread('pictures/scene1.ppm');
img1 = imresize(img1,[680 850]);
img2 = imread('pictures/FD8.JPG');
% img2 = imread('pictures/scene5.ppm');
img2 = imresize(img2,[680 850]);

% initialisation
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

% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end


%% extract interest points & their descriptors
[F1_raw,D1_raw]=vl_sift(single(img1));
[F2_raw,D2_raw]=vl_sift(single(img2));
coords1_sift_raw = horzcat(F1_raw(2,:)',F1_raw(1,:)');
coords2_sift_raw = horzcat(F2_raw(2,:)',F2_raw(1,:)');

%% sort by confidence in descending order
[~,idx1] = sort(F1_raw(4,:),'descend');
[~,idx2] = sort(F2_raw(4,:),'descend');

%% get the strongest interest points and their descriptors
if size(F1_raw,2) > num_strongest
    F1 = F1_raw(:,idx1(1:num_strongest));
    F2 = F2_raw(:,idx2(1:num_strongest));
    D1 = D1_raw(:,idx1(1:num_strongest));
    D2 = D2_raw(:,idx2(1:num_strongest));
    coords1_sift = coords1_sift_raw(idx1(1:num_strongest),:);
    coords2_sift = coords2_sift_raw(idx2(1:num_strongest),:);
    points1_sift = horzcat(coords1_sift,F1(3:4,:)');
    points2_sift = horzcat(coords2_sift,F2(3:4,:)');
else
   F1 = F1_raw;
   F2 = F2_raw;
   D1 = D1_raw;
   D2 = D2_raw;
   coords1_sift = coords1_sift_raw;
   coords2_sift = coords2_sift_raw;
   points1_sift = horzcat(coords1_sift,F1(3:4,:)');
   points2_sift = horzcat(coords2_sift,F2(3:4,:)');
end


%% visualise the strongest N points in both images
if visualise
    figure;
    subplot(2,2,1);
    myvisualise(points1_sift, img1, 'very original corner image 1');
    subplot(2,2,2);
    myvisualise(points2_sift, img2, 'very original corner image 2');
end

%% Matching correspondence (our own implementation)
[matchedPoints1_sift, matchedPoints2_sift] = myFeatureMatching(...
            coords1_sift, coords2_sift, D1', D2', knn_threshold);

%% Visualise matched correspondences before RANSAC
if visualise
    figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
    h=title(ax, 'Matched Correspondences before RANSAC');
    set(h,'FontSize',16);
    legend(ax, 'Matched points 1','Matched points 2');
end


%% Compute Fundamental Matrix using RANSAC
numOfTrials = 100000;
thr = 0.01;
[bestf1, f, inliers1] = RansacPredictF(matchedPoints1_sift,...
                      matchedPoints2_sift,numOfTrials,thr);
                  
[bestf2, f, inliers2] = RansacPredictF(matchedPoints2_sift,...
                      matchedPoints1_sift,numOfTrials,thr);

matchedPoints1_inliers = [];
matchedPoints2_inliers = [];
cnt = 1;
for i=1:size(matchedPoints1_sift,1)
    if (inliers1(i) == 1)
        matchedPoints1_inliers(cnt,:) = matchedPoints1_sift(i,:);
        matchedPoints2_inliers(cnt,:) = matchedPoints2_sift(i,:);
        cnt = cnt +1;
    end
end
if visualise
    f=figure;
    drawEpipolarline(bestf1, matchedPoints1_sift, img2);
    h=title('Img1 - Epipolar Lines and Epipoles');
    set(h,'FontSize',30);
end

% Calculate epipole
[isIn,epipole1] = isEpipoleInImage(bestf1,size(img1))
[isIn,epipole2] = isEpipoleInImage(bestf2,size(img2))

