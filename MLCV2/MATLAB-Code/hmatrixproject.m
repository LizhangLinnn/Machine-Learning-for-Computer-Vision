function projection = hmatrixproject(toBeProjected, besth)
% INPUT
% =========================================================================
% toBeProjected : The coordinates of points to be projected in image 1
% besth         : The best homography matrix onbtained from RansacPredictH
%
% OUTPUT
% =========================================================================
% projection : the coordinates of projected points
x_size = size(toBeProjected,1);
for i = 1:x_size
   tmp=besth*[toBeProjected(i,:),1]'; 
   q = tmp(3,:);
   projection(i,:) = [tmp(1,:)./q tmp(2,:)./q];
end
end