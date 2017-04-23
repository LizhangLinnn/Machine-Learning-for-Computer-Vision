% Simple Random Forest Toolbox for Matlab
% written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% updated by Tae-Kyun Kim, Feb 09, 2017

% This is a guideline script of simple-RF toolbox.
% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox

% Under BSD Licence

% Initialisation
init;


%%
% Test on the dense 2D grid data, and visualise the results ... 

% Change the RF parameter values and evaluate ... 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% experiment with Caltech101 dataset for image categorisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
% Set 'normalised' in getData.m to 1/0 to enable/disable normalisation on the bag-of-words histograms

t_start = tic;
accuracyAndTime_Kvalue = {};
% for K_i = 1:2%Kvalues % search for the value of K
%     Kvalues = [256 512];
%     K = Kvalues(K_i);
K = 256;
accuracyAndTime_norm = {};
% for loop = 1:2 %1:5
%     for normalised_i = 1:2 % search for the effect of normalisation
%     normalised = [0 1];
%     normalised = normalised(normalised_i);
    normalised = 0;
    showImg = 0;
    showHist = 0;
    [data_train, data_test, Test_Quant_Time] = getData('Caltech',normalised,showImg,showHist,K); %data, normalised, show image, K value for K-means
    % time elapsed.
    %% RF classifier
    % Kvalue
    % normalised/unnormalised
    % calculating average value
    % training and testing time cost.
    accuracyAndTime_numOfTrees = {};
        for num_i = 1:5 % search for the number of trees
            accuracyAndTime_depth = {};
            for depth_i = 1:5
                accuracyAndTime_DoR = {};
                for DoR_i = 1:7
                    accuracyAndTime_stopprob = {};
                    for stopprob_i = 4 %1:5
                        accuracyAndTime_classID = {};
                        for classID_i = 1:4 %1:5
                            accuracyAndTime_emptypercentage = [];
                            for emptypercentage_i = 3 %1:5
                                emptypercentage = [0.025 0.045 0.074 0.09 0.11];
                                emptypercentage = emptypercentage(emptypercentage_i);
                                classID = 1:5; % test class one only for now..
                                classID = classID(classID_i);
                                depth = 5:9;
                                depth = depth(depth_i);
                                DoR = [2 5 10 30 50 100 300];
                                DoR = DoR(DoR_i);
                                stopprob = [0.4 0.55 0.7 0.8 0.95];
                                stopprob = stopprob(stopprob_i);
                                num = [10 30 50 70 90];
                                num = num(num_i);
                                % Set the random forest parameters ...
                                % These codes are copied from my coursework partner's Github - matianci111.
                                visualise = 0;
                                RFparam.num = num; %100;         % Number of trees
                                RFparam.splitNum = DoR;     % Degree of randomness
                                RFparam.split = 'IG';     % IGNORE THIS, NOT USEFUL
                                RFparam.classID = classID %1; % 4 is distancelearner, the other three are shown in the weakTrain.m file
                                % stopping criteria
                                RFparam.depth = depth %8;        % trees depth
                                RFparam.emptypercentage = emptypercentage %0.04; % percentage of emptiness
                                RFparam.stopprob = stopprob %0.8; % probability density

                                % Train Random Forest ...
                                train_start = tic;
                                trees = growTrees(data_train, RFparam);
                                time_train = toc(train_start);


                                % Evaluate/Test Random Forest ...
                                test_start = tic;
                                correct_count = 0;
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
                                time_test = toc(test_start);

                                % show accuracy and confusion matrix ...
                                [~,idx] = max(p_rf_sum,[],2);
                                accuracy_test = correct_count/size(data_test,1);



                                % Evaluate/Train Random Forest ...
                                test_train_start = tic;
                                correct_count = 0;
                                for n=1:size(data_test,1)
                                    classIDX = ceil(n/15);
                                    %disp(sprintf('testing the %d th test data',classIDX))
                                    %leaves stand for which leave does a single test point come out from
                                    leaves = testTrees([data_train(n,1:end-1) 0],trees);
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
                                time_test_train = toc(test_train_start);

                                % show accuracy and confusion matrix ...
                                [~,idx] = max(p_rf_sum,[],2);
                                accuracy_train = correct_count/size(data_train,1);


                                accuracyAndTime_emptypercentage(emptypercentage_i,:) = [accuracy_test , accuracy_train, time_train, time_test];
                            end % end emptypercentage search
                            accuracyAndTime_classID{classID_i,1} = accuracyAndTime_emptypercentage;
                        end % end classID search
                        accuracyAndTime_stopprob{stopprob_i,1} = accuracyAndTime_classID;
    %                     accuracyAndTime_stopprob{stopprob_i,1} = accuracyAndTime_emptypercentage;
                    end
                    accuracyAndTime_DoR{DoR_i,1} = accuracyAndTime_stopprob;
                end % end stopprob search
                accuracyAndTime_depth{depth_i,1} = accuracyAndTime_DoR; 
                
            end % end depth search
            accuracyAndTime_numOfTrees{num_i,1} = accuracyAndTime_depth;
        end % end number of trees search
%         accuracyAndTime_norm{normalised_i,loop} = accuracyAndTime_numOfTrees;
%     end % end normalisation search
% end % end averaging loop
%     accuracyAndTime_Kvalue{K_i,1} = accuracyAndTime_norm;
% end % end K value search
End = toc(t_start);
% 
% %% plot results of simulation (with varying stoping criterion)
% Kvalues = [256 512];
% normalised = [0 1];
% num = [20 60 100];
% depth = 5:9;
% stopprob = 0.6:0.04:0.9;
% emptypercentage = [0.02 0.034 0.047 0.06 0.074];
% 
% for K = 1: length(Kvalues)
% for norm = 1:length(normalised)
% figure;
% suptitle(sprintf('Kvalues = %d, normalised = %d',Kvalues(K), normalised(norm)));
% for n =1:length(num)
% for d =1:length(depth)
%     acc = [];
%     for s = 1:length(stopprob)
%         acc = horzcat(acc,accuracyAndTime_Kvalue{K,1}{norm,1}{n,1}{d,1}{s,1}(:,1));
%     end
%     subplot(length(num),length(depth),(n-1)*length(depth)+d);
%     ma = max(max(acc));
%     mi = min(min(acc));
%     surf(stopprob,emptypercentage,acc);
%     zlim([mi ma]);
%     xlabel('stopprob');
%     ylabel('emptypercentage');
%     title(sprintf('number of trees %d, tree depth %d',num(n),depth(d)));
%     
% end
% end
% end
% end
% 
% %% plot results of simulation (averaged by various stoping criterion)
% Kvalues = [256 512];
% normalised = [0 1];
% num = [20 60 100];
% depth = 5:9;
% stopprob = 0.6:0.04:0.9;
% emptypercentage = [0.02 0.034 0.047 0.06 0.074];
% 
% figure;
% for K = 1: length(Kvalues)
% for norm = 1:length(normalised)
%     num_accuracy = [];
% for n =1:length(num)
%     depth_accuracy = [];
% for d =1:length(depth)
%     acc = [];
%     for s = 1:length(stopprob)
%         acc = horzcat(acc,accuracyAndTime_Kvalue{K,1}{norm,1}{n,1}{d,1}{s,1}(:,1));
%     end
%     sizeOfacc = size(acc,1) * size(acc,2);
%     sumOfacc = sum(sum(acc));
%     ave_accuracy = sumOfacc / sizeOfacc;
%     depth_accuracy(d,:) = ave_accuracy;
% end
%     num_accuracy = horzcat(num_accuracy,depth_accuracy(:));
% end
%     subplot(length(Kvalues),length(normalised),(K-1)*length(normalised)+norm);
%     surf(num,depth,num_accuracy);
%     xlabel('Trees');
%     ylabel('Depth');
%     title(sprintf('Kvalues = %d, normalised = %d',Kvalues(K), normalised(norm)));
% end
% 
% 
% end
% 
% %% plot results of simulation (fixed stop criteria)
% Kvalues = [256 512];
% normalised = [0 1];
% num = [20 60 100];
% depth = 5:9;
% stopprob = 0.6:0.04:0.9;
% emptypercentage = [0.02 0.034 0.047 0.06 0.074];
% 
% figure;
% for K = 1: length(Kvalues)
% for norm = 1:length(normalised)
%     num_accuracy = [];
% for n =1:length(num)
%     depth_accuracy = [];
% for d =1:length(depth)
% %     for s = 1:length(stopprob)
%     s = 8;
%     e = 5;
%     acc = accuracyAndTime_Kvalue{K,1}{norm,1}{n,1}{d,1}{s,1}(e,1);
% %     end
%     depth_accuracy(d,:) = acc;
% end
%     num_accuracy = horzcat(num_accuracy,depth_accuracy(:));
% end
%     subplot(length(Kvalues),length(normalised),(K-1)*length(normalised)+norm);
%     surf(num,depth,num_accuracy);
%     xlabel('Trees');
%     ylabel('Depth');
%     title(sprintf('Kvalues = %d, normalised = %d\n stopprob = %.2f, emptypercentage = %.1f%%',Kvalues(K), normalised(norm),stopprob(s),100*emptypercentage(e)));
% end
% 
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% % random forest codebook for Caltech101 image categorisation
% % .....