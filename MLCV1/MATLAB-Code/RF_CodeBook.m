%% RF codebook
num = 100;
depth = 5; %7:9;
splitNum = 30;
classID = 1;%1:3;
emptypercentage = 0.08; %[0.003 0.01];
stopprob = 0.3;

data1_num = {};
for num_i = 1: length(num)
    data_depty = {};
    for depth_i = 1: length(depth)
        data3_splitNum = {};
        for splitNum_i = 1: length(splitNum)
            data4_classID = {};
            for classID_i = 1:length(classID)
                data5_emptypercent={};
                for emptypercentage_i = 1:length(emptypercentage)
                    data6_stopprob = {};
                   for stopprob_i = 1:length(stopprob)
    normalised = 0;
    showImg = 1;
    showHist = 0;
    param.num = num(num_i); %100;         % Number of trees
    param.splitNum = depth(depth_i);     % Degree of randomness
    param.split = 'IG';     % IGNORE THIS, NOT USEFUL
    param.classID = classID_i; %1; % 4 is distancelearner, the other three are shown in the weakTrain.m file
    % stopping criteria
    param.depth = depth(depth_i); %8;        % trees depth
    param.emptypercentage = emptypercentage(emptypercentage_i); %0.04; % percentage of emptiness
    param.stopprob = stopprob(stopprob_i); %0.8; % probability density
    time_start = tic;
    [data_train, data_test, Test_Quant_Time, codeBookTrees] = getDataCodeBook(normalised,showImg,showHist,param); %data, normalised, show image, K value for K-means
    time_codebook = toc(time_start);
                    data6_stopprob{stopprob_i,1} = data_train;
                    data6_stopprob{stopprob_i,2} = data_test;
                    data6_stopprob{stopprob_i,3} = codeBookTrees;
                    data6_stopprob{stopprob_i,4} = Test_Quant_Time;
                    data6_stopprob{stopprob_i,5} = time_codebook;
                   end
                    data5_emptypercent{emptypercentage_i,1} = data6_stopprob;
                end
                data4_classID{classID_i,1} = data5_emptypercent;
            end
            data3_splitNum{splitNum_i,1} = data4_classID;
        end
        data2_depth{depth_i,1} = data3_splitNum;
    end
    data1_num{num_i,1} = data2_depth;
end
    %% RF classifier
    % Kvalue
    % normalised/unnormalised
    % calculating average value
    % training and testing time cost.
    
% Set the random forest parameters ...
% These codes are copied from my coursework partner's Github - matianci111.
loops = 50;
accuracy_test = zeros(loops,1);
t_loop = tic;
for loop = 1:loops
visualise = 0;
RFparam.num = 150; %100;         % Number of trees
RFparam.splitNum = 500 ;     % Degree of randomness
RFparam.split = 'IG';     % IGNORE THIS, NOT USEFUL
RFparam.classID = 1; %1; % 4 is distancelearner, the other three are shown in the weakTrain.m file
% stopping criteria
RFparam.depth = 7; %8;        % trees depth
RFparam.emptypercentage = 0.08; %0.04; % percentage of emptiness
RFparam.stopprob = 0.6; %0.8; % probability density

% Train Random Forest ...
train_start = tic;
trees = growTrees(data_train, RFparam);
time_train = toc(train_start);


% Evaluate/Test Random Forest ...
test_start = tic;
correct_count = 0;
p_rf_sum = [];
    for n=1:size(data_test,1)
        classIDX = ceil(n/15);
        %disp(sprintf('testing the %d th test data',classIDX))
        %leaves stand for which leave does a single test point come out from
        leaves = testTrees([data_test(n,:) 0],trees);
        % average the class distributions of leaf nodes of all trees
        p_rf = trees(1).prob(leaves,:);
        p_rf_sum(n,:) = sum(p_rf)/length(trees);


        % visualise
        if visualise
            figure;
            visualise_leaf_with_label(trees, leaves);
            subplot(ceil(size(leaves,2)^0.5),ceil(size(leaves,2)^0.5),11);
            bar(p_rf_sum(n,:));
            axis([0 10.5 0 1]);
            title('averaged data distribution');
            suptitle(sprintf('test image from class %d',classIDX));
        end

        % predicted group labels
        [~,idx] = max(p_rf_sum(n,:));
        if (idx == classIDX)
            correct_count = correct_count + 1;
        end


    % C = confusionmat(,idx)
    % impact of the vocabulary size on classification accuracy

    % impact of normalisation on classification accuracy

    end % end testing

% show accuracy and confusion matrix ...
[~,idx] = max(p_rf_sum,[],2);
accuracy_test(loop,1) = correct_count/size(data_test,1);
end
accuracy_test_ave = sum(accuracy_test)/length(accuracy_test)
t_loop = toc(t_loop)

% % Evaluate/Train Random Forest ...
% test_train_start = tic;
% correct_count = 0;
% for n=1:size(data_test,1)
%     classIDX = ceil(n/15);
%     %disp(sprintf('testing the %d th test data',classIDX))
%     %leaves stand for which leave does a single test point come out from
%     leaves = testTrees([data_train(n,1:end-1) 0],trees);
%     % average the class distributions of leaf nodes of all trees
%     p_rf = trees(1).prob(leaves,:);
%     p_rf_sum(n,:) = sum(p_rf)/length(trees);
% 
% 
%     % visualise
%     if visualise
%         figure;
%         visualise_leaf_with_label(trees, leaves);
%         subplot(ceil(size(leaves,2)^0.5),ceil(size(leaves,2)^0.5),11);
%         bar(p_rf_sum(n,:));
%         axis([0 10.5 0 1]);
%         title('averaged data distribution');
%         suptitle(sprintf('test image from class %d',classIDX));
%     end
% 
%     % predicted group labels
%     [~,idx] = max(p_rf_sum(n,:));
%     if (idx == classIDX)
%         correct_count = correct_count + 1;
%     end
% 
% 
% % C = confusionmat(,idx)
% % impact of the vocabulary size on classification accuracy
% 
% % impact of normalisation on classification accuracy
% 
% end % end testing
% time_test_train = toc(test_train_start);
% [~,idx] = max(p_rf_sum,[],2);
% accuracy_train = correct_count/size(data_train,1);


% show accuracy and confusion matrix ...
% accuracy = [accuracy_test, time_train, time_test];

