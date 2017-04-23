%% Q2_1_C using SIFT
% Estimate homography from different number of correspondences from Q1.2 
% starting from the minimum number up to the maximum number of available 
% pairs. Report and discuss HA for different number of correspondences. 
% Find the number of outliers in your list of automatic correspondences 
% and explain your approach to that.

%% parameters to be adjusted
knn_threshold = 1.4;
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


%% extract interest points & their descriptors
[F1_raw,D1_raw]=vl_sift(single(img1));
[F2_raw,D2_raw]=vl_sift(single(img2));
coords1_sift_raw = horzcat(F1_raw(2,:)',F1_raw(1,:)');
coords2_sift_raw = horzcat(F2_raw(2,:)',F2_raw(1,:)');

%% sort by confidence in descending order
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
    subplot(121);
    myvisualise(points1_sift, img1, 'very original corner image 1');
    subplot(122);
    myvisualise(points2_sift, img2, 'very original corner image 2');
end

%% Matching correspondence
[matchedPoints1_sift, matchedPoints2_sift] = myFeatureMatching(coords1_sift,...
                coords2_sift, D1', D2', knn_threshold);
%% visualise matched correspondences before RANSAC
if visualise
    figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1_sift),fliplr(matchedPoints2_sift),'montage','Parent',ax);
    title(ax, 'our own implementation');
    legend(ax, 'Matched points 1','Matched points 2');
end

%% Compute ground truth inliers and outliers
numberoftrial = 10000;
threshold = 2; % average pixels

[besth, f2, inliersIndex] = RansacPredictH(matchedPoints1_sift, matchedPoints2_sift, numberoftrial, threshold);
matchedPoints1_inliers_ground_truth = [];
matchedPoints2_inliers_ground_truth = [];
cnt = 1;
cnt_out = 1;
outliers1 = [];
outliers2 = [];
% obtain coordinates of inliers and outliers
for i=1:size(matchedPoints1_sift,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers_ground_truth(cnt,:) = matchedPoints1_sift(i,:);
        matchedPoints2_inliers_ground_truth(cnt,:) = matchedPoints2_sift(i,:);
        cnt = cnt +1;
    else
        outliers1(cnt_out,:) = matchedPoints1_sift(i,:);
        outliers2(cnt_out,:) = matchedPoints2_sift(i,:);
        cnt_out = cnt_out + 1;
    end
end
projectedv2 = hmatrixproject(matchedPoints1_inliers_ground_truth, besth);

%% visualise ground truth
if visualise
    f_auto=figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers_ground_truth),...
        fliplr(matchedPoints2_inliers_ground_truth),'montage','Parent',ax);
    h = title(ax, 'Ground Truth');
    set(h,'FontSize',22);
    legend(ax, 'Matched points 1','Matched points 2');
    % print(f_auto,'-dpng','-r600');
end

%% visualise Outliers obtained when computing ground truth
if visualise
    f_auto=figure; ax = axes;
    showMatchedFeatures(img1,img2,fliplr(outliers1),...
        fliplr(outliers2),'montage','Parent',ax);
    h = title(ax, 'Ground Truth');
    set(h,'FontSize',22);
    legend(ax, 'Matched points 1','Matched points 2');
    % print(f_auto,'-dpng','-r600');
end


%% Find the relationship between the number of correspondence and HA
% params
numberoftrial = 1000;
threshold = 6.7; % average pixels
num_loop = 100;
step = 1;
num_points = 4:step:size(matchedPoints1_sift,1);
% num_points = 4:step:25;
num_step = size(num_points,2);

% init
HA = zeros(num_loop,num_step);

for n_step = 1:num_step
for iter = 1:num_loop
[rand_points1,idx1] = datasample(matchedPoints1_sift,num_points(n_step),1,'Replace',false);
rand_points2 = matchedPoints2_sift(idx1,:);

[testh, f2, inliersIndex] = RansacPredictH(rand_points1, rand_points2, numberoftrial, threshold);

matchedPoints1_inliers = [];
matchedPoints2_inliers = [];
cnt = 1;
for i=1:size(rand_points1,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers(cnt,:) = rand_points1(i,:);
        matchedPoints2_inliers(cnt,:) = rand_points2(i,:);
        cnt = cnt +1;
    end
end


projectedv2 = hmatrixproject(matchedPoints1_inliers_ground_truth, testh);



% Compute HA error
HA(iter,n_step) = calculateHA(projectedv2, matchedPoints2_inliers_ground_truth);
num_outliers(iter,n_step) = num_points(n_step) - size(matchedPoints1_inliers,1);
end
end
HA_ave_auto = mean(HA,1);
HA_std_auto = std(HA,0,1);


%% visualise average HA error against number of correspondences
if visualise
    f = figure; 
    [a,h1,h2]=plotyy(num_points, HA_ave_auto, num_points, HA_std_auto);
    xlim(a,[num_points(1) num_points(end)]);
    xlabel('Number of Correspondances',...
        'FontWeight','Bold','FontSize',24);
    ylabel(a(1),'Average HA (pixels)',...
        'FontWeight','Bold','FontSize',24);
    ylabel(a(2),'Standard Deviation',...
        'FontWeight','Bold','FontSize',24);
    set(a(1),'FontSize',20);
    set(a(2),'FontSize',20);
    set(a, 'XTick', num_points(1):2:num_points(end));
    y1 = ylim(a(1)); y2 = ylim(a(2));
    set(a(1), 'YTick', linspace(y1(1),y1(2),5));
    set(a(2), 'YTick', linspace(y2(1),y2(2),5));
    set(h1,'LineWidth',3,'LineStyle','--');
    set(h2,'LineWidth',3,'LineStyle',':');
    set(f, 'Units', 'normalized', 'Position', [0.2,0.2,0.8,0.8]);
    title('HA against the NO. of Correspondences');
    legend('Averaged HA', 'Standard Deviation');
    % print(f,'-dpng','-r600');
end
