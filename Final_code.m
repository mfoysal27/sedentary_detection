clc
clear all;
%% 
addpath(genpath(pwd));
obj = VideoReader('Test_2.mp4');
no_frame=ceil(obj.FrameRate*obj.Duration);
map=zeros(1,int8(no_frame));
i=1;
% Database=zeros(144, 108,no_frame-1 );
for k = 1 : no_frame-1   
  I = imresize(readFrame(obj), 1);

  figure(1)
   imshow(I);
%   imwrite(I,strcat('Image_test2_', num2str(k), '.jpg'));
% imwrite(I,strcat('Image_2_', num2str(k), '.jpg'));
  map(1, i)=sum(sum(sum(I(:, :, :))))/15552;
  i=i+1;
  Database(:, :, k)=I(:, :,1);
%   I=createMask(I);
%   I=imgaussfilt(rgb2gray(I), .2);
I2=imresize(I, .25);
I2=im2bw(I2);
  I2=edge (I2);

  I2=mat2gray(I2);
    imshow(I2); hold on;
    I2= imread('C:\Users\mfoysal\OneDrive - Texas Tech University\Desktop\Spring 19\Conference Paper\kamrul poster\edge2.tif');
    I2=(I2(:, :, 1));
        imshow(I2); hold on;
  points =detectSURFFeatures(I2);
plot(points.selectStrongest(20));
points.Location(1, :)

end
% map=sgolay(map, 3, 10);
plot(map)



%%

% I=im2uint8(imread('sitting2.jpg'));
% Iblur = imgaussfilt(I,8);
% x = 0:size(Iblur,2)-1;
% y = 0:size(Iblur,1)-1;
% [X,Y] = meshgrid(x,y);
% meshc(X, Y, Iblur(:, :, 1));
% hold on;

%
imds = imageDatastore('Images','IncludeSubfolders',true,'LabelSource','foldernames');

% numberOfImages = length(imds.Files)
% for k = 1 : numberOfImages
%   inputFileName = imds.Files{k};
%   rgbImage = imread(inputFileName);
%   grayImage = edge(rgb2gray(rgbImage));
%   imwrite(grayImage, imds.Files{k});
% %   imshow(rgbImage);
% end





tbl = countEachLabel(imds);
[trainingSet, validationSet] = splitEachLabel(imds, 0.6, 'randomize');
bag = bagOfFeatures(trainingSet);
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
% confMatrix = evaluate(categoryClassifier, trainingSet);
% confMatrix = evaluate(categoryClassifier, validationSet);
% mean(diag(confMatrix))

%% Testing
for i=1:20
    fullfile=strcat('Testing Data\1 (', num2str(i), ').jpg');
%     I_test=imread(fullfile);
    I_test=edge(rgb2gray(imread(fullfile)));
%   points =detectSURFFeatures(uint8());
%  plot(points.selectStrongest(5))
% im = im2uint8(imread(I_test));
 im = im2uint8((I_test));
[labelIdx, scores] = predict(categoryClassifier, im);
i
% Display the string label
categoryClassifier.Labels(labelIdx)
end
%%
% c = kmeans(I(:), 3);

%%
% imds = imageDatastore('Images','IncludeSubfolders',true,'LabelSource','foldernames');
% 
% [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end
% 
% %%
% net=alexnet;
% layers=net.Layers;
% layers(1)= imageInputLayer([108, 144, 3], 'Name', 'Input');
% inputSize = layers(1).InputSize;
% 
% layersTransfer = layers(1:end-9);
% numClasses = numel(categories(imdsTrain.Labels))
% 
% 
% layers_alex = [
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% 
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',10, ...
%     'MaxEpochs',8, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augimdsValidation, ...
%     'ValidationFrequency',3, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% netTransfer = trainNetwork(augimdsTrain,layers_alex,options);
% [YPred,scores] = classify(netTransfer,augimdsValidation);
% 
% idx = randperm(numel(imdsValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label));
% end
% 
% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)













