function [matchedPoints1, matchedPoints2] = myFeatureMatching(C1,C2,D1,D2,threshold)
% INPUT:
% ========================================================================
% C1        : a list of coordinates of detected interest points in image A
% C2        : a list of coordinates of detected interest points in image B
% D1        : feature descriptors of detected interest points in image A
% D2        : feature descriptors of detected interest points in image B
% threshold : the ratio between 1-NN (first NN) and 2-NN (second NN)
%
% OUTPUT:
% ========================================================================
% matchedPoints1 : coordinates of matched interest points in image A
% matchedPoints2 : coordinates of matched interest points in image B

    x_size = size(D1,1);
    y_size = size(D2,1);
    for i = 1:x_size
        tmp_distance = [];
        for j = 1:y_size
            tmp_distance(j) = sum(abs(D1(i,:)-D2(j,:)))/y_size;
        end
        
        [value, idx] = sort(tmp_distance,'ascend');
        min_v = value(1);

        % 1-NN to 2-NN ratio threshold
        snd_min_v = value(2);
        if min_v > (snd_min_v/threshold)
            min_v = inf; %indicating corresponding descriptor not matched
        end

        distance(i) = min_v;
        index(i) = idx(1);
    end
    
    
    
    % Now filter out matches with distance inf
    [distance,idx] = sort(distance,'ascend');
    matchedIP_idx = idx(~isinf(distance)); % the index of matched interest points

    idx_matchedIP_img2_temp = index;
    idx_matchedIP_img1_temp = 1:1:size(index,2);
    idx_matchedIP_img1 = idx_matchedIP_img1_temp(matchedIP_idx);
    idx_matchedIP_img2 = idx_matchedIP_img2_temp(matchedIP_idx);

    % remove correspondance that are not matched uniquely (which are ambiguous)
    if threshold ~= 1
        arr1 = getNonRepeatableElementIdx(idx_matchedIP_img1);
        arr2 = getNonRepeatableElementIdx(idx_matchedIP_img2);
        idx = intersect(arr1,arr2);
        correspondance = [idx_matchedIP_img1(idx);idx_matchedIP_img2(idx)];
        if size(correspondance,2)==0
           error('user defined error - no interest point is matched under current parameters'); 
        end
    else
        correspondance = [idx_matchedIP_img1;idx_matchedIP_img2];
    end
    matchedPoints1=C1(correspondance(1,:),:);
    matchedPoints2=C2(correspondance(2,:),:);

end

function [idx] = getNonRepeatableElementIdx(input_array)
% this function returns the index of non-repeatable elements in the input
% array
[u,idx_u,~] = unique(input_array,'stable');
[u_sorted,idx_u_sorted] = sort(u,'ascend');
temp = histc(input_array,u_sorted);
[~, idx_nonRepeat] = find(temp==1);
idx = idx_u(idx_u_sorted(idx_nonRepeat));
idx = sort(idx,'ascend');
end