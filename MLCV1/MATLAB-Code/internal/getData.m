function [ data_train, data_query, Test_Quant_Time] = getData( MODE ,normalise,showImg,showHist,K)
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
           
        PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
        PHOW_Step_train = 8; % The lower the denser. Select from {2,4,8,16}
        %% note here we can later further dense the test PHOW_Step to 4 or 2.
        PHOW_Step_test = 8; % The lower the denser. Select from {2,4,8,16}

        %% get descriptors of training and testing images 
        
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(4:end).name}; % 10 classes
        
        disp('Loading training images...')
        
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % list of all images in the current folder
            imgIdx{c} = randperm(length(imgList)); % randomise the order of images without replacement (numbers do not repeat)
            imgIdx_tr(c,:) = imgIdx{c}(1:imgSel(1)); % select the first 15 images for training
            imgIdx_te(c,:) = imgIdx{c}(imgSel(1)+1:sum(imgSel)); % select the 16th to 30th images for testing
            
            
            for i = 1:size(imgIdx_tr,2)
                
                I_train = imread(fullfile(subFolderName,imgList(imgIdx_tr(c,i)).name)); 
                I_test = imread(fullfile(subFolderName,imgList(imgIdx_te(c,i)).name)); 
                
                % if the image is of 3 dimensions (colour image)
                if size(I_train,3) == 3
                    I_train = rgb2gray(I_train); % PHOW work on gray scale image
                end
                if size(I_test,3) == 3
                    I_test = rgb2gray(I_test); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                % convert I to single precision to boost the processing speed
                [~, desc_tr{c,i}] = vl_phow(single(I_train),'Sizes',PHOW_Sizes,'Step',PHOW_Step_train); %  extracts PHOW features (multi-scaled Dense SIFT)
                [~, desc_te{c,i}] = vl_phow(single(I_test),'Sizes',PHOW_Sizes,'Step',PHOW_Step_test); %  extracts PHOW features (multi-scaled Dense SIFT)
                
            end
        end

        %% Build visual vocabulary (codebook) for 'Bag-of-Words method'

        disp('Building visual codebook...')
        
        %visual vocabulary
        visual_Vocabulary = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        
        %% K-means clustering 
        disp('K-means clustering...') 
        % number of codewords in the codebook
        numBins = K; 
        % centers: row(number of clusters) x column(dimensions of a codeword)
        [~, centers] = kmeans(visual_Vocabulary',numBins,'Distance','sqeuclidean','Replicates',1);
        
        % After k-means clustering, we have numBins number of clusters and 
        % their corresponding cluster centers.
        
        
        
        %% Training image quantisation and Obtaining Bag-of-words Histograms
        disp('Encoding training Images...')
        % use classifiers to assign the descriptors of the training image
        % to the nearest cluster center. Then pile up at the corresponding
        % column in the histogram.
        numOfClasses = 10;
        numOfImgsForDisplay = 15; %display the bag-of-words histogram of training images per selected class
        
        for j=1:numOfClasses % loop across different image classes
            subFolderName = fullfile(folderName,classList{j});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % list of all images in the current folder
            
            if (showHist || showImg)
                figure;
                suptitle(sprintf('class %d',j));
            end
            for i=1:numOfImgsForDisplay % images in the same class
                
                % replace the descriptor
                train_descriptor = desc_tr{j,i};
                idxes_train{j,i} = knnsearch(centers,train_descriptor');
                idx = idxes_train{j,i};
                
                % matrix that stores histogram data
                if normalise
                    trainHistData(j*15+i-15,:) = histcounts(idx,numBins)/size(desc_tr{j,i},2);
                else
                    trainHistData(j*15+i-15,:) = histcounts(idx,numBins);
                end
                
                % Visualise and compare images before and after vector quantisation
                % Visualise
                if (showImg) % visualise images in the same class and their bag-of-word hist 
                    subplot(ceil(numOfImgsForDisplay/3),6,2*i-1);
                    I = imread(fullfile(subFolderName,imgList(imgIdx_tr(j,i)).name)); 
                    imshow(I);
                    drawnow;
                    % plot bag of words histograms of training images
                    subplot(ceil(numOfImgsForDisplay/3),6,2*i);
                    histogram(idx,0.5:(numBins+0.5));
                    xlim([0 numBins+1])
                else
                    if (showHist) % only visualise hist of images in the same class
                    subplot(ceil(numOfImgsForDisplay/3),3,i);
                    histogram(idx,0.5:(numBins+0.5));
                    xlim([0 numBins+1])
                    end
                end
            end
        end
        
        %% Testing image quantisation and Obtaining Bag-of-words Histograms
        disp('Encoding testing Images...')
        numOfClasses = 10;
        numOfImgsForDisplay = 15; %display the bag-of-words histogram of training images per selected class
        %showTestImg = 0;
        
        tic;
        t_start = tic;
        testHistData = trainHistData;
        for j=1:numOfClasses % loop across different image classes
            subFolderName = fullfile(folderName,classList{j});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % list of all images in the current folder
            
            if (showHist || showImg)
                figure;
                suptitle(sprintf('class %d',j));
            end
            for i=1:numOfImgsForDisplay % images in the same class
                
                % replace the descriptor
                test_descriptor = desc_te{j,i};
                idxes_test{j,i} = knnsearch(centers,test_descriptor');
                idx = idxes_test{j,i};
                
                % matrix that stores histogram data
                if normalise
                    testHistData(j*15+i-15,:) = histcounts(idx,numBins)/size(desc_te{j,i},2);
                else
                    testHistData(j*15+i-15,:) = histcounts(idx,numBins);
                end
                
                % Visualise
                if (showImg ==1) % visualise images in the same class and their bag-of-word hist 
                    subplot(ceil(numOfImgsForDisplay/3),6,2*i-1);
                    I_test = imread(fullfile(subFolderName,imgList(imgIdx_te(j,i)).name));
                    imshow(I_test);
                    drawnow;
                    % plot bag of words histograms of training images
                    subplot(ceil(numOfImgsForDisplay/3),6,2*i);
                    histogram(idx,0.5:(numBins+0.5));
                    xlim([0 numBins+1])
                else
                    if (showHist) % only visualise hist of images in the same class
                    subplot(ceil(numOfImgsForDisplay/3),3,i);
                    histogram(idx,0.5:(numBins+0.5));
                    xlim([0 numBins+1])
                    end
                end
            end
        end
        toc
        Test_Quant_Time = toc(t_start);
        
        %% output training and testing data
        classlabel = [];
        for i = 1:10
            classlabel = vertcat(classlabel,ones(15,1)*i);
        end
        data_train = horzcat(trainHistData, classlabel);
        data_query = testHistData;
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
        
        

end

