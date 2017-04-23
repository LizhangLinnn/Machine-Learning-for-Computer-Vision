% 1) Manual
boat1 = imread('pictures/scene1.ppm');
boat2 = imread('pictures/scene2.ppm');
imshow(boat1);
[x,y] = ginput(1);
x1 = x;
y1 = y;
figure;
imshow(boat2);
[x,y] = ginput(1);
x2 = x;
y2 = y;