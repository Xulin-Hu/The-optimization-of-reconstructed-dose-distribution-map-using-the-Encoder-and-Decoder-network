%% Author:Xulin Hu
%  Date: 2025.5.8
%  Demo function: The denoise of traditional reconstruction algorithms (TR、DAS、BP etc...)
% using the 'Encoder-Decoder' network

clear all;
% % Step 1: Load the datasets of TR algorithm reconstruction
% XA pressure datasets of TR algorithm reconstruction
load('.\dataset\Reconstructed_XA_Pressure\XTrain_Bone.mat'); % or load '.\dataset\XTrain_Bone.mat';
load('.\dataset\Reconstructed_XA_Pressure\XTrain_Soft_Tissue.mat');
load('.\dataset\Reconstructed_XA_Pressure\XTrain_Tissue_equivalent_plastic.mat');
XTrain = [XTrain_Bone,XTrain_Soft_Tissue,XTrain_Tissue_equivalent_plastic]; % Noisy images as input

% Ground truth datasets for XA pressure
load('.\dataset\Ground_Truth\XTrain_Bone_GT.mat'); % or load '.\dataset\XTrain_Bone.mat';
load('.\dataset\Ground_Truth\XTrain_Soft_Tissue_GT.mat');
load('.\dataset\Ground_Truth\XTrain_Tissue_equivalent_plastic_GT.mat');
YTrain = [XTrain_Bone_GT,XTrain_Soft_Tissue_GT,XTrain_Tissue_equivalent_plastic_GT]; % Clean images as ground truth

% Step 2: Data preprocessing
% Let's assume the training data is a set of pairs of TR reconstruction and Ground truth images.
% We choose 120 images to pretrain the model.

% % 假设 inputImage3 是 N×1 或 1×N 的元胞数组，每个元素是一个图像
XTrain = cellfun(@(x) imresize(x, [64 64]), XTrain, 'UniformOutput', false);
YTrain = cellfun(@(x) imresize(x, [64 64]), YTrain, 'UniformOutput', false);

% noisyImage1 = imresize(noisyImage1, [64 64]);   % Inputimage is 120 x 120 
% inputImage1 = imresize(inputImage1, [64 64]);   % Inputimage is 120 x 120
N = 19; % N为训练集样本数
XTrain_samples = XTrain(1:N); % Noisy images as input
YTrain_samples = XTrain(1:N); % Clean images as ground truth

% Step 3: U-Net Architecture Definition
inputSize = [64 64 1]; % 需要修改网络架构，满足120*120图片输入（解决数据集问题）
numClasses = 1; % For RGB output

% Encoder (Downsampling Path)
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    % Downsampling blocks
    convolution2dLayer(3,64,'Padding','same','Name','conv1_1')
    reluLayer('Name','relu1_1')
    convolution2dLayer(3,64,'Padding','same','Name','conv1_2')
    reluLayer('Name','relu1_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool1')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv2_1')
    reluLayer('Name','relu2_1')
    convolution2dLayer(3,128,'Padding','same','Name','conv2_2')
    reluLayer('Name','relu2_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool2')

    convolution2dLayer(3,256,'Padding','same','Name','conv3_1')
    reluLayer('Name','relu3_1')
    convolution2dLayer(3,256,'Padding','same','Name','conv3_2')
    reluLayer('Name','relu3_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool3')

    convolution2dLayer(3,512,'Padding','same','Name','conv4_1')
    reluLayer('Name','relu4_1')
    convolution2dLayer(3,512,'Padding','same','Name','conv4_2')
    reluLayer('Name','relu4_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool4')

    % Bottleneck
    convolution2dLayer(3,1024,'Padding','same','Name','conv5_1')
    reluLayer('Name','relu5_1')
    convolution2dLayer(3,1024,'Padding','same','Name','conv5_2')
    reluLayer('Name','relu5_2')
    
    % Upsampling Path (Decoder)
    transposedConv2dLayer(2,512,'Stride',2,'Cropping','same','Name','upconv4')
    convolution2dLayer(3,512,'Padding','same','Name','conv6_1')
    reluLayer('Name','relu6_1')
    convolution2dLayer(3,512,'Padding','same','Name','conv6_2')
    reluLayer('Name','relu6_2')
    
    transposedConv2dLayer(2,256,'Stride',2,'Cropping','same','Name','upconv3')
    convolution2dLayer(3,256,'Padding','same','Name','conv7_1')
    reluLayer('Name','relu7_1')
    convolution2dLayer(3,256,'Padding','same','Name','conv7_2')
    reluLayer('Name','relu7_2')
    
    transposedConv2dLayer(2,128,'Stride',2,'Cropping','same','Name','upconv2')
    convolution2dLayer(3,128,'Padding','same','Name','conv8_1')
    reluLayer('Name','relu8_1')
    convolution2dLayer(3,128,'Padding','same','Name','conv8_2')
    reluLayer('Name','relu8_2')
    
    transposedConv2dLayer(2,64,'Stride',2,'Cropping','same','Name','upconv1')
    convolution2dLayer(3,64,'Padding','same','Name','conv9_1')
    reluLayer('Name','relu9_1')
    convolution2dLayer(3,64,'Padding','same','Name','conv9_2')
    reluLayer('Name','relu9_2')

    % Output layer
    convolution2dLayer(1, numClasses, 'Name', 'conv10')
    regressionLayer('Name','output') % For regression since it's a denoising task
];

% Step 4: Define Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

% Step 5: Train the U-Net Model
net = trainNetwork(cat(4, XTrain_samples{:}), cat(4, YTrain_samples{:}), layers, options);

% Step 6: Evaluate the Model on the Test Image
% 取最后一张图片作为测试数据
OriginalImage = cell2mat(YTrain(N+1));
NoisyImage = cell2mat(XTrain(N+1)); 

denoisedImage = predict(net, NoisyImage); 

% Step 7: Display Results
figure;
% imshowpair(noisyImage, denoisedImage, 'montage');
% title('Noisy Image (Left) vs Denoised Image (Right)');

subplot(1,3,1); imshow(OriginalImage); title('Original Image');
subplot(1,3,2); imshow(NoisyImage); title('Noisy Image');
subplot(1,3,3); imshow(denoisedImage); title('denoised Image');
colormap parula;

% Step 8: Display the reconstructed relative errors
Reconstruction_dose = reshape(denoisedImage, [], 1); % 转换为列向量
Ground_truth = reshape(OriginalImage, [], 1); % 转换为列向量
[mae1,mse1,rmse1,mape1,error1,errorPercent1,mre1]=calc_error(Ground_truth,Reconstruction_dose);

