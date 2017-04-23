% Set the random forest parameters ...
% These codes are copied from my coursework partner's Github - matianci111.
loops = 50;
accuracy_test = zeros(loops,1);
t_loop = tic;
for loop = 1:loops
visualise = 0;
RFparam.num = 100; %100;         % Number of trees
RFparam.splitNum = 100;     % Degree of randomness
RFparam.split = 'IG';     % IGNORE THIS, NOT USEFUL
RFparam.classID = 1; %1; % 4 is distancelearner, the other three are shown in the weakTrain.m file
% stopping criteria
RFparam.depth = 8; %8;        % trees depth
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