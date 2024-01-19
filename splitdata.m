clc;clear;close all
%%
% Percobaan 90:10
im =  imageDatastore('/Users/macbookpro/Downloads/dataskripsi/training','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(@(loc)),[224,224]);
[Train ,Test] = splitEachLabel(im,0.9,'randomized');
save('90Train.mat');
%%
% Percobaan 80:20
im =  imageDatastore('/Users/macbookpro/Downloads/dataskripsi/training','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[Train ,Test] = splitEachLabel(im,0.8,'randomized');
save('80Train.mat');
%%
% Percobaan 70:30
im =  imageDatastore('/Users/macbookpro/Downloads/dataskripsi/training','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[Train ,Test] = splitEachLabel(im,0.7,'randomized');
save('70Train.mat');

% Percobaan 60:40
im =  imageDatastore('/Users/macbookpro/Downloads/dataskripsi/training','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[Train ,Test] = splitEachLabel(im,0.6,'randomized');
save('60Train.mat');