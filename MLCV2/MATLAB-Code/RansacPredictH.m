function [besth, h, inliers] = RansacPredictH(m1, m2, numberoftrial, threshold)
% INPUT
% =========================================================================
% m1 : Coordinates of matched interest points in image 1. The columns are
%      coordinates and the rows are different interest points
%
% m2 : Coordinates of matched interest points in image 2. The columns are
%      coordinates and the rows are different interest points
%
% numberoftrail : The number of RANSAC iterations
%
% threshold     : RANSAC threshold. In our implementation, it stands for
%                 the Manhatten distance between the matched points in 
%                 image 2 and the projected correspondence.
%
% OUTPUT
% =========================================================================
% besth   : The best homography matrix produced by RANSAC iterations
%
% inliers : Coordinates of points that are within the pre-specified 
%           threshold when transformation is performed by the besth
%
% h       : Homography matrix obtained by using method in slide 20 with
%           inliers calculated from RANSAC

    if size(m1,1) < 4
        error('given fewer than 4 points');
    end
    newm1 = [];
    newm2 = [];
    [inliers(:), besth] = myRansac(m1, m2, numberoftrial, threshold);
    for i = 1:size(inliers, 2)
       if(inliers(i)==1)
        newm1(end+1,:) = m1(i,:);
        newm2(end+1,:) = m2(i,:);
       end
    end
    h = gethmatrix(newm1, newm2);
end

function [inliers, besth] = myRansac(m1, m2, numberoftrial, mythreshold)
    N_Trials = numberoftrial;
    threshold = mythreshold;
    Max_N_Inliers = -1;
    besth = zeros(3,3);
    for i = 1:N_Trials
        [distance, h] = myEstTFormDistance(m1, m2);
        [Inliers, N_Inliers] = findInliers(distance, threshold);
        
        if Max_N_Inliers < N_Inliers
            Max_N_Inliers = N_Inliers;
            inliers = Inliers;
            besth = h;
        end
    end
end

function [d, h] = myEstTFormDistance(m1, m2)
    sizeofindex = size(m1,1);
    indices = generateRandomIndices(sizeofindex);
    h = gethmatrix(m1(indices,:), m2(indices,:));
    d = computeDistance(m1, m2, h);
end



function [indices] = generateRandomIndices(sizeofindex)
    indices = zeros(4,1);
    for i = 1:4
        uniqueflag = false;
        while ~uniqueflag
            tmp = randi(sizeofindex);
            if(sum(find(indices == tmp))==0)
                uniqueflag = true;
            end
        end
        indices(i) = tmp;
    end
end

function d = computeDistance(m1, m2, h)
     projectedv2 = hmatrixproject(m1, h);
     d = sum(abs(projectedv2-m2),2);
end

function [inliers, N_Inliers] = findInliers(distance, threshold)
    
    inliers = [];
    N_Inliers = 0;

    for i = 1: size(distance, 1)
        if (distance(i) <= threshold)
            inliers(i) = true;
            N_Inliers = N_Inliers + 1;
        else
            inliers(i) = false;
        end
    end
end

function h = gethmatrix(m1, m2)
%3A) obtain homography matrix
n = size(m1, 1);
% Solve equations using SVD
x1 = m1(:,1)';
y1 = m1(:,2)';
x2 = m2(:,1)';
y2 = m2(:,2)';
rowofzero = zeros(3, n);
rowofxy = -[x1; y1; ones(1,n)];
hx = [rowofxy; rowofzero; x2.*x1; x2.*y1; x2];
hy = [rowofzero; rowofxy; y2.*x1; y2.*y1; y2];
h = [hx hy];
[U, D, V] = svd(h');
h = (reshape(V(:,9), 3, 3)).';
end
