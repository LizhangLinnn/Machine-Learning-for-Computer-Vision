% Q_2_2_e
% e) Change the focal length by 2mm, repeat Q2.2.d and compare. 
% Add small random noise (e.g. Gaussian with max 2 pixel) to the disparity 
% map, repeat Q2.2.d and compare.

% Field of view, Depth of field and Perspective

%% Load the Parameters of the Stereo Camera
% Note that the parameters used here are pre-calculated using Matlab
% Stereo Camera Calibration APP
load('stereoParams.mat');

%% Read and Rectify Frames
img1 = imread('pictures/FD8.JPG');
% img1 = imread('pictures/scene1.ppm');
img2 = imread('pictures/FD7.JPG');
% img2 = imread('pictures/scene2.ppm');


% zoom the images by 2mm focal length (from 29mm to 31mm)
img_height = size(img1,1);
img_length = size(img1,2);
delta_h = img_height*2/31;
delta_l = img_length*2/31;

img1 = img1(1+delta_h:end-delta_h,1+delta_l:end-delta_l,:);
img2 = img2(1+delta_h:end-delta_h,1+delta_l:end-delta_l,:);
img1 = imresize(img1,[680 850]);
img2 = imresize(img2,[680 850]);


[img1, img2] = ...
    rectifyStereoImages(img1, img2, stereoParams);

figure;
imshow(stereoAnaglyph(img1, img2));
title('Rectified Image','FontSize',16);

%% Compute Disparity
% In rectified stereo images any pair of corresponding points are located 
% on the same pixel row. For each pixel in the left image compute the
% distance to the corresponding pixel in the right image. This distance is
% called the disparity, and it is proportional to the distance of the
% corresponding world point from the camera.
frameLeftGray  = rgb2gray(img1);
frameRightGray = rgb2gray(img2);
disparity_range = [0 64];
disparityMap = disparity(frameLeftGray, frameRightGray,...
                    'DisparityRange',disparity_range);

figure;
imshow(disparityMap,disparity_range);
title('Disparity Map (without noise, f=31mm)','FontSize',16);
colormap jet
colorbar

%% Reconstruct the 3-D Scene (depth map)
% Reconstruct the 3-D world coordinates of points corresponding to each
% pixel from the disparity map.
points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', img1);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
xlabel(player3D.Axes, 'Horizontal Direction (m)','FontSize',14);
ylabel(player3D.Axes, 'Vertical Direction (m)','FontSize',14);
zlabel(player3D.Axes, 'Depth (m)','FontSize',14);
title(player3D.Axes, 'Depth Map (without Noise, f=31mm)','FontSize',16);

% Visualize the point cloud
view(player3D, ptCloud);
% title('Depth Map (Larger Focal Length)');

%% Add small random noise to disparity map and repeat plotting depth map
disparityMap_noisy = disparityMap + (rand(size(disparityMap,1),size(disparityMap,2))-0.5).*4;
figure;
imshow(disparityMap_noisy, disparity_range);
title('Disparity Map (with noise, f=31mm)','FontSize',16);
colormap jet
colorbar

%% depth map
points3D_noisy = reconstructScene(disparityMap_noisy, stereoParams);

% Convert to meters and create a pointCloud object
points3D_noisy = points3D_noisy ./ 1000;
ptCloud_noisy = pointCloud(points3D_noisy, 'Color', img1Rect);

% Create a streaming point cloud viewer
player3D_noisy = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
xlabel(player3D_noisy.Axes, 'Horizontal Direction (m)','FontSize',14);
ylabel(player3D_noisy.Axes, 'Vertical Direction (m)','FontSize',14);
zlabel(player3D_noisy.Axes, 'Depth (m)','FontSize',14);
title(player3D_noisy.Axes, 'Depth Map (with Noise, f=31mm)','FontSize',16);

% Visualize the point cloud
view(player3D_noisy, ptCloud_noisy);
