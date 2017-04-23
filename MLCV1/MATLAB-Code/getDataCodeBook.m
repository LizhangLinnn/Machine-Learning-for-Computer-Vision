function [ data_train, data_query, Test_Quant_Time, trees] = getDataCodeBook(normalise,showImg,showHist,param)
% Generate RF Codebook and process vector quantisation
close all;



PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step_train = 16; % The lower the denser. Select from {2,4,8,16}
% note here we can later further dense the test PHOW_Step to 4 or 2.
PHOW_Step_test = 16; % The lower the denser. Select from {2,4,8,16}

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
        % add image category label to both training image descriptors
        classLabel = c*ones(1,size(desc_tr{c,i},2));
        desc_tr{c,i} = [desc_tr{c,i};classLabel];
    end
end

%% get visual vocabulary (codebook) for 'Bag-of-Words method'

disp('Building visual codebook...')

%visual vocabulary
visual_Vocabulary = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering

%% growing visual codewords

trees = growTrees(visual_Vocabulary',param);


%% Training image quantisation and Obtaining Bag-of-words Histograms
disp('Encoding training Images...')
% use classifiers to assign the descriptors of the training image
% to the nearest cluster center. Then pile up at the corresponding
% column in the histogram.
numOfClasses = 10;
numOfImgsForDisplay = 15; %display the bag-of-words histogram of training images per selected class
trainHistData = [];
train_start = tic;
for j=1:numOfClasses % loop across different image classes 

    subFolderName = fullfile(folderName,classList{j});
    imgList = dir(fullfile(subFolderName,'*.jpg')); % list of all images in the current folder

    if (showHist || showImg)
        figure;
        suptitle(sprintf('class %d',j));
    end
    for i=1:numOfImgsForDisplay % images in the same class

        % get the descriptors of the image being quantised
%         rand_select100 = randperm(size(desc_tr{j,i},2),100);
%         train_descriptor = (desc_tr{j,i}(:,rand_select100))';
        train_descriptor = desc_tr{j,i}';
        numOfDes = size(train_descriptor,1);
        numBins = size(trees(1).prob,1);
        
        % quantising
        leaves = zeros(1,numOfDes*length(trees));
        for n = 1:numOfDes
%             leaves(1,((n-1)*numOfdes+1):(n*numOfdes)) = testTrees([train_descriptor(n,1:end-1) 0],trees); 
            leaves(1,((n-1)*length(trees)+1):(n*length(trees))) = testTrees_fast(train_descriptor(n,1:end-1),trees);
        end
        
        % matrix that stores histogram data
        if normalise
            %Note here the indexes of the leaves must be in the range of numBins
            trainHistData(j*15+i-15,:) = histcounts(leaves,numBins)/numOfDes;
        else
            %Note here the indexes of the leaves must be in the range of numBins
            trainHistData(j*15+i-15,:) = histcounts(leaves,numBins);
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
            histogram(leaves,0.5:(numBins+0.5));
            xlim([0 numBins+1])
        else
            if (showHist) % only visualise hist of images in the same class
            subplot(ceil(numOfImgsForDisplay/3),3,i);
            drawnow;
            histogram(leaves,0.5:(numBins+0.5));
            xlim([0 numBins+1])
            end
        end
    end
end
train_end = toc(train_start);

%% Testing image quantisation and Obtaining Bag-of-words Histograms
disp('Encoding testing Images...')
% use classifiers to assign the descriptors of the training image
% to the nearest cluster center. Then pile up at the corresponding
% column in the histogram.
numOfClasses = 10;
numOfImgsForDisplay = 15; %display the bag-of-words histogram of training images per selected class

testHistData = [];
t_start = tic;

for j=1:numOfClasses % loop across different image classes 

    subFolderName = fullfile(folderName,classList{j});
    imgList = dir(fullfile(subFolderName,'*.jpg')); % list of all images in the current folder

    if (showHist || showImg)
        figure;
        suptitle(sprintf('class %d',j));
    end
    for i=1:numOfImgsForDisplay % images in the same class

        % get the descriptors of the image being quantised
%         rand_select100 = randperm(size(desc_te{j,i},2),100);
%         test_descriptor = (desc_te{j,i}(:,rand_select100))';
        test_descriptor = desc_te{j,i}';
        numOfDes = size(test_descriptor,1);
        numBins = size(trees(1).prob,1);
        
        % quantising
        leaves = zeros(1,numOfDes*length(trees));
        for n = 1:numOfDes
%             leaves(1,n:(n*numOfdes)) = testTrees_fast([test_descriptor(n,:) 0],trees);
            leaves(1,((n-1)*length(trees)+1):(n*length(trees))) = testTrees_fast(test_descriptor(n,:),trees);
        end
        
        % matrix that stores histogram data
        if normalise
            %Note here the indexes of the leaves must be in the range of numBins
            testHistData(j*15+i-15,:) = histcounts(leaves,numBins)/numOfDes;
        else
            %Note here the indexes of the leaves must be in the range of numBins
            testHistData(j*15+i-15,:) = histcounts(leaves,numBins);
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
            histogram(leaves,0.5:(numBins+0.5));
            xlim([0 numBins+1])
        else
            if (showHist) % only visualise hist of images in the same class
            subplot(ceil(numOfImgsForDisplay/3),3,i);
            histogram(leaves,0.5:(numBins+0.5));
            xlim([0 numBins+1])
            end
        end
    end
end
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




