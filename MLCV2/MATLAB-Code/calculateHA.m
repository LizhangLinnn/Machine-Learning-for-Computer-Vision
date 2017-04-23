function HA = calculateHA(v1, v2)
% INPUT
% =========================================================================
% v1 : a set of projections from image 1 to image 2
% v2 : the matched points in image 2, which corresponds to the points that
%      are to be projected from image 1 to image 2
% OUTPUT
% =========================================================================
% HA : Homography error (weighted sum of cityblock distance)

ydif = sum(abs(v1(:,1)-v2(:,1)))/size(v1,1);
xdif = sum(abs(v1(:,2)-v2(:,2)))/size(v1,1);
HA = ydif+xdif;
end