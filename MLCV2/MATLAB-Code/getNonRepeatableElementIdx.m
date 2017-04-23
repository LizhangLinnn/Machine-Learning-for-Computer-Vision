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