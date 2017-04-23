function [bestf, f, inliers] = RansacPredictF(m1, m2, numberoftrial, threshold)
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
% threshold : threshold for determining inliers 
%
% OUTPUT
% =========================================================================
% bestf   : The best fundamental matrix produced by RANSAC iterations
%
% inliers : Coordinates of points that are within the pre-specified 
%           threshold when transformation is performed by the bestf
%
% f       : Fundamental matrix obtained by using Eight-Point Algorithm with
%           inliers calculated from RANSAC

    if size(m1,1) < 8
        error('given fewer than 8 points');
    end
    newv1 = [];
    newv2 = [];
    [inliers(:), bestf] = myRansac(m1, m2, numberoftrial, threshold);
    for i = 1:size(inliers, 2)
       if(inliers(i)==1)
        newv1(end+1,:) = m1(i,:);
        newv2(end+1,:) = m2(i,:);
       end
    end
    f = getfmatrix(newv1, newv2);
end

function [inliers, bestf] = myRansac(v1, v2, numberoftrial, mythreshold)
    N_Trials = numberoftrial;
    threshold = mythreshold;
    Max_N_Inliers = -1;
    bestf = zeros(3,3);
    for i = 1:N_Trials
        [distance, f] = myEstTFormDistance(v1, v2);
        [Inliers, N_Inliers] = findInliers(distance, threshold);
        
        if Max_N_Inliers < N_Inliers
            Max_N_Inliers = N_Inliers;
            inliers = Inliers;
            bestf = f;
        end
    end
end

function [d, f] = myEstTFormDistance(v1, v2)
    sizeofindex = size(v1,1);
    indices = generateRandomIndices(sizeofindex);
    f = getfmatrix(v1(indices,:), v2(indices,:));
    d = computeDistance(v1, v2, f);
end



function [indices] = generateRandomIndices(sizeofindex)
    indices = zeros(8,1);
    for i = 1:8
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

function d = computeDistance(v1, v2, f)
    v1h = zeros(size(v1,1),3);
    v2h = zeros(size(v2,1),3);
    v1h(:, 3) = 1;
    v1h(:, 1:2) = v1;
    v2h(:, 3)   = 1;
    v2h(:, 1:2)  = v2;
    
    epipole = (v2h * f)';
    tmp = epipole .* v1h';
    d = sum(tmp, 1) .^ 2;
    
    %sampson error
    epl1 = f * v1h';
    epl2 = f' * v2h';
    d = d ./ (epl1(1,:).^2 + epl1(2,:).^2 + epl2(1,:).^2 + epl2(2,:).^2);
end

function [inliers, N_Inliers] = findInliers(distance, threshold)
    
    inliers = zeros(size(distance,2),1);
    N_Inliers = 0;

    for i = 1: size(distance, 2)
        if (distance(i) <= threshold)
            inliers(i) = true;
            N_Inliers = N_Inliers + 1;
        else
            inliers(i) = false;
        end
    end
end

function f = getfmatrix(v1, v2)
%3B) obtain fundamental matrix
n = size(v1, 1);
A = [v1(:,1).*v2(:,1) v1(:,1).*v2(:,2) v1(:,1) ...
      v1(:,2).*v2(:,1) v1(:,2).*v2(:,2) v1(:,2) ...
      v2(:,1) v2(:,2) ones(n,1)];
F = zeros(3,3);
[UA,SA,VA]=svd(A);
F(:) = VA(:,9);
[U,S,V] = svd(F);
f = U(:,1:2)*S(1:2,1:2)*V(:,1:2)';
end