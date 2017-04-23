function [node,nodeL,nodeR] = splitNode(data,node,param, T, maxdepth, nodeindex)
% Split node

visualise = 1;
labels = 1:10;
% Initilise child nodes
nodeL = struct('idx',[],'model',struct,'splitfun',0,'isleaf',0,'prob',[]);
nodeR = struct('idx',[],'model',struct,'splitfun',0,'isleaf',0,'prob',[]);

if length(node.idx) == 0
    return;
end

if length(node.idx) <= param.emptysize % make this node a leaf if has less than 5 data points stopping criteria
    node.isleaf = 1;
    return;
end

prob = reshape(histc(data(node.idx,end),labels),[],1);
prob = prob/sum(prob);
for i = 1:size(prob,1)
    if prob(i) >= param.stopprob
        node.isleaf = 1;
        return;
    end
end

idx = node.idx;
data = data(idx,:);
X = data(:,1:end-1);
Y = data(:,end);

%initialize parameters, classifiercommitfirst is unavailable
opts.classifierID = param.classID;
opts.numSplits = param.splitNum;
opts.classifierCommitFirst = true;
index = [];
bestgain=0;
finalclassifier =0;
model = struct;

%train to split the data using all 4 split functions
%{
opts.classifierID = 1;
[model1, bestgain1, final_index1, iglist1] = weakTrain(X, Y, opts);

opts.classifierID = 2;
[model2, bestgain2, final_index2, iglist2] = weakTrain(X, Y, opts);

opts.classifierID = 3;
[model3, bestgain3, final_index3, iglist3] = weakTrain(X, Y, opts);
%}
[model4, bestgain4, final_index4, iglist4] = weakTrain(X, Y, opts);
index = final_index4;
finalclassifier=param.classID;
model = model4;
bestgain = bestgain4;

node.splitfun = finalclassifier;
node.model = model;

if nodeindex > 2^(maxdepth-2)-1
    if ~isempty(idx(index))
        nodeL.isleaf = 1;
    end
    if ~isempty(idx(~index))
        nodeR.isleaf = 1;
    end
end

if isempty(idx(index)) || isempty(idx(~index))
    node.splitfun = 0;
    node.isleaf = 1;
    return;
end

nodeL.idx = idx(index);
nodeR.idx = idx(~index);

%if visualise
%    visualise_splitfunc(final_index1,data,model1.r,model1.t,bestgain1,0)
%    fprintf('Information gain = %f. \n',bestgain1);
%end

end