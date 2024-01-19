clc;clear;close all
%%
% IMPORT DATA
a = load('90Train.mat');
Train = a.Train;
Test = a.Test;

% NETWORK
net = mobilenetv2;
ly = net.Layers;
prob = 0.9;
%ly(141) = dropoutLayer(prob);

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = numel(categories(Train.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

miniBatchSize = 8;
valFrequency = floor(numel(Train.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',3e-4, ...      
    'Shuffle','every-epoch', ...
    'ValidationData',Test, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% TRAINING
[newnet,info] = trainNetwork(Train,lgraph,options);
%%
% TESTING
[predict,scores] = classify(newnet,Test);
names = Test.Labels;
pred = (predict==names);
analyzeNetwork(net)
figure, plotconfusion(names,predict)

save('80_P09_B8_DS')