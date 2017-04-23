function [] = drawEpipolarline(f, v, img)
% INPUT
% =========================================================================
% f   : fundamental matrix
% v   : inliers of matched correspondences 
%       row     -  different correspondences
%       column  -  x and y coordinates
% img : the image that the plot is based on
%
% OUTPUT
% =========================================================================
% HA : Homography error (weighted sum of cityblock distance)

imagesc(img), axis ([0 size(img,2) 0 size(img,1)]), colormap(gray), hold on
epiLines = epipolarLine(f,v);
points = lineToBorderPoints(epiLines,[size(img,2),size(img,1)]);
line(points(:,[2,4])',points(:,[1,3])');
plot(v(:,2),v(:,1),'ro','LineWidth',2); 
truesize;
hold off
end