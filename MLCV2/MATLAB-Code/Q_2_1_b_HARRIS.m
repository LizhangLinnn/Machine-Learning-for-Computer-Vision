% Experimental results
% Q2.1.b

% read images
img1 = imread('pictures/HG1.jpg');
img1 = imresize(img1,[680 850]);
img2 = imread('pictures/HG2.jpg');
img2 = imresize(img2,[680 850]);

% convert colour scale to gray scale
% img 1
if (size(size(img1),2) == 3)
    img1 = rgb2gray(img1);
end
% img 2
if (size(size(img2),2) == 3)
    img2 = rgb2gray(img2);
end



%% paramaters that can be adjusted
threshold = 0.01; % 1% of the maximum HARRIS response value 
visualise = 1;
knn_threshold = 1.04;
num_strongest = 500; % number of strongest interest points
ANMS = true; % adaptive non-maximal suppression
radius = 2;
descriptor_type = 'CIRCLE';
descriptor_patch_diameter = 32; % diameter of circle descriptor


%% Get automatic interest points and features

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
    descriptor1 = getDescriptor(points1(:,1:2), img1, descriptor_patch_diameter);
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
    descriptor2 = factor*getDescriptor(points2(:,1:2), img2, descriptor_patch_diameter/factor);
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


%% Visualise Interest Points before matching
if visualise
    f1=figure;
    subplot(1,2,1);
    myvisualise(points1,img1,'Img1 - Interest Points');
    subplot(1,2,2);
    myvisualise(points2,img2,'Img2 - Interest Points')
    set(f1, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     print(f1,'-dpng','-r600');
end

%% Matching correspondance 
[matchedPoints1, matchedPoints2] = myFeatureMatching(points1(:,1:2),points1(:,1:2),descriptor1,descriptor2,knn_threshold);

%% visualise matched correspondences
if visualise
figure; ax = axes;
showMatchedFeatures(img1,img2,fliplr(matchedPoints1),fliplr(matchedPoints2),'montage','Parent',ax);
title(ax, 'Matched Correspondances before RANSAC (HARRIS)');
legend(ax, 'Matched points 1','Matched points 2');
end

%% Q2_1_a - 
% Compare the interest points obtained in these two cases using HA error.
% =======================================================================
numberoftrial = 10000;
threshold = 3; % cityblock distance in pixels
[besth, f2, inliersIndex] = RansacPredictH(matchedPoints1, matchedPoints2, numberoftrial, threshold);

matchedPoints1_inliers1 = [];
matchedPoints2_inliers1 = [];
cnt = 1;
for i=1:size(matchedPoints1,1)
    if (inliersIndex(i) == 1)
        matchedPoints1_inliers1(cnt,:) = matchedPoints1(i,:);
        matchedPoints2_inliers1(cnt,:) = matchedPoints2(i,:);
        cnt = cnt +1;
    end
end

projectedv2 = hmatrixproject(matchedPoints1_inliers1, besth);

% Compute HA error
HA = calculateHA(projectedv2, matchedPoints2_inliers1);

%%  visualise matched correspondences after RANSAC
if ~visualise
f_auto=figure;
subplot(121);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers1),...
    fliplr(matchedPoints2_inliers1));
% showMatchedFeatures(img1,img2,fliplr(points1_manual_inliers),fliplr(points2_manual_inliers),'montage','Parent',ax);
title('Using Square Descriptor','FontSize',20);
h=legend('Matched points 1','Matched points 2');
set(h,'FontSize',16)

subplot(122);
showMatchedFeatures(img1,img2,fliplr(matchedPoints1_inliers2),...
    fliplr(matchedPoints2_inliers2));
% showMatchedFeatures(img1,img2,fliplr(points1_manual_inliers),fliplr(points2_manual_inliers),'montage','Parent',ax);
title('Using Circle Descriptor','FontSize',20);
h=legend('Matched points 1','Matched points 2');
set(h,'FontSize',16)
% print(f_auto,'-dpng','-r600');
end