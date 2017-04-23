% Q_2_2_f_c_d
% f) rectify images
% c) Calculate disparity map between Image A and B
% d) Plot 3D depth map

%% Load the Parameters of the Stereo Camera
% Note that the parameters used here are pre-calculated using Matlab
% Stereo Camera Calibration APP
load('stereoParams.mat');

%% Read and Rectify Frames
img1 = imread('pictures/FD8.JPG');
img2 = imread('pictures/FD7.JPG');

img2 = imresize(img2,[680 850]);
img1 = imresize(img1,[680 850]);

figure;
subplot(121);
imshow(img1);
subplot(122);
imshow(img2);

%% Rectification (f)
[img1Rect, img2Rect] = ...
    rectifyStereoImages(img1, img2, stereoParams);

f = figure;
subplot(121);
imshow(stereoAnaglyph(img1Rect, img2Rect));
title('Rectified Left and Right Images','FontSize',16);

%% Compute Disparity (c)
% In rectified stereo images any pair of corresponding points are located 
% on the same pixel row. For each pixel in the left image compute the
% distance to the corresponding pixel in the right image. This distance is
% called the disparity, and it is proportional to the distance of the
% corresponding world point from the camera.
img1Gray  = rgb2gray(img1Rect);
img2Gray = rgb2gray(img2Rect);
    
disparity_range = [0 64];
disparityMap = disparity(img1Gray, img2Gray,'DisparityRange',disparity_range);

subplot(122);
imshow(disparityMap, disparity_range);
title('Disparity Map (f=29mm)','FontSize',16);
colormap jet
colorbar

%% Reconstruct the 3-D Scene (d)
% Reconstruct the 3-D world coordinates of points corresponding to each
% pixel from the disparity map.
points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', img1Rect);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
xlabel(player3D.Axes, 'Horizontal Direction (m)','FontSize',14);
ylabel(player3D.Axes, 'Vertical Direction (m)','FontSize',14);
zlabel(player3D.Axes, 'Depth (m)','FontSize',14);
title(player3D.Axes, 'Depth Map (f=29mm)','FontSize',16);

% Visualize the point cloud
view(player3D, ptCloud);
