%% Q2_1_b using SIFT
%% parameters to be adjusted
knn_threshold = 1.4;
num_strongest = 500; % number of strongest interest points
visualise = 1;
num_manual_IPs = 10; % number of manually selected interest points
%% read image
% img1 = imread('pictures/img1.pgm');
% img2 = imread('pictures/img2.pgm');
img1 = imread('pictures/HG1.JPG');
img2 = imread('pictures/HG2.JPG');

img1 = imresize(img1,[680 850]);
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
img1 = im2double(img1);
% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end
img2 = im2double(img2);


%% Get Manual interest points and features
% [points1_manual, points2_manual] = getManualIPs(img1,img2,num_manual_IPs);
load lizhang_manual_points.mat

if visualise
    f_manual = figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(points1_manual),...
        fliplr(points2_manual),'montage','Parent',ax);
    h = title('Manually Matched Points');
    set(h,'FontSize',22);
    set(f_manual, 'Units', 'normalized', 'Position', [0,0,0.6,0.6]);
    % print(f_manual,'-dpng','-r600');
end

% estimate homography matrix and HA
numberoftrial = 1000;
threshold = 6.7; % pixels - 6.7 is chosen such that all manually selected
                 % points are inliers
[besth_manual, f2, inliersIndex] = RansacPredictH(points1_manual, points2_manual, numberoftrial, threshold);

points1_manual_inliers = [];
points2_manual_inliers = [];
cnt = 1;
for i=1:size(points1_manual,1)
    if (inliersIndex(i) == 1)
        points1_manual_inliers(cnt,:) = points1_manual(i,:);
        points2_manual_inliers(cnt,:) = points2_manual(i,:);
        cnt = cnt +1;
    end
end

projectedv2 = hmatrixproject(points1_manual, besth_manual);

% Compute HA error
HA = calculateHA(projectedv2, points2_manual);


%%  visualise matched correspondences after RANSAC (Manual)
if visualise
    f_auto_manual=figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(points1_manual_inliers),...
        fliplr(points2_manual_inliers),'montage','Parent',ax);
    title(ax, 'Matched Correspondences after RANSAC - Manual');
    legend(ax, 'Matched points 1','Matched points 2');
    % print(f_auto,'-dpng','-r600');
end





% ========================================================================
% ========================================================================
% ========================================================================
% ========================================================================
% =========================Here Goes Automatic============================
% =========================Here Goes Automatic============================
% =========================Here Goes Automatic============================
% =========================Here Goes Automatic============================
% ========================================================================
% ========================================================================
% ========================================================================
% ========================================================================
% ========================================================================

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

%% visualise the strongest N points in both images
if visualise
    figure;
    subplot(1,2,1);
    imshow(img1);
    hold on ;
    vl_plotframe(F1);
    title('Interest Points (SIFT)');
    subplot(1,2,2);
    imshow(img2);
    hold on ;
    vl_plotframe(F2);
    title('Interest Points (SIFT)');
end

%% Matching correspondence
[matchedPoints1_sift, matchedPoints2_sift] = myFeatureMatching(coords1_sift,...
                coords2_sift, D1', D2', knn_threshold);

%% visualise matched correspondences
if visualise
figure; 
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage');
title('Matched Correspondences');
legend('Matched points 1','Matched points 2');
end

%% Compute homography matrix 
% Using 10 matched correspondences, in order to be consistent with the 
% number manual points. The result is averaged

numberoftrial = 10000;
threshold = 6.7; % average pixels
num_loop = 10;

% init
HA = zeros(num_loop,1);

for iter = 1:num_loop
[rand_points1,idx1] = datasample(matchedPoints1_sift,size(points1_manual,1),1,'Replace',false);
rand_points2 = matchedPoints2_sift(idx1,:);

[besth_auto, f2, inliersIndex] = RansacPredictH(rand_points1, rand_points2, numberoftrial, threshold);

matchedPoints1_inliers = [];
matchedPoints2_inliers = [];
cnt = 1;
for i=1:size(points1_manual,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers(cnt,:) = rand_points1(i,:);
        matchedPoints2_inliers(cnt,:) = rand_points2(i,:);
        cnt = cnt +1;
    end
end

projectedv2 = hmatrixproject(matchedPoints1_inliers, besth_auto);

% Compute HA error
HA(iter,1) = calculateHA(projectedv2, matchedPoints2_inliers);
end
HA_ave_auto = mean(HA);

%%  visualise matched correspondences after RANSAC
if visualise
f_auto_manual=figure; 
subplot(121);
showMatchedFeatures(img2,img2,fliplr(matchedPoints2_inliers),...
    fliplr(projectedv2),'PlotOptions',{'ro','g+','y-'});
title('Auto - Projected Inliers vs. Matched Inliers','FontSize',22);
l1 = legend('Inliers in Image 2', 'Projection');
set(l1,'FontSize',16);

subplot(122);
projectedv2_manual = hmatrixproject(matchedPoints1_inliers, besth_manual);
showMatchedFeatures(img2,img2,fliplr(matchedPoints2_inliers),...
    fliplr(projectedv2_manual),'PlotOptions',{'ro','g+','y-'});
title('Manual - Projected Inliers vs. Matched Inliers','FontSize',22);
l2=legend('Inliers in Image 2','Projection');
set(l2,'FontSize',16);
set(f_auto_manual, 'Units', 'normalized', 'Position', [0,0,1,1]);
% print(f_auto_manual,'-dpng','-r600');
end