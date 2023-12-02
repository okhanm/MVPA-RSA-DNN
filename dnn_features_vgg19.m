%% DNN-vgg19: extracting features

function [deepnn] = dnn_features_vgg19

% Load Data

results_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\results_DNN\');
images = imageDatastore('stim','LabelSource','foldernames');

numImages = numel(images.Labels);
idx = 1:numImages;
figure (1)

for i = 1:size(idx,2) % ploting stimuli
    subplot(3,9,i)
    I = readimage(images,idx(i));
    imshow(I);
end

% DNN
dnn = vgg19;
dnnName = 'vgg19';

net = dnn;

% Analyze the network architecture. The first layer, the image input layer, requires input images of size 224-by-224-by-3, where 3 is the number of color channels.
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net);

% Extract Image Features
augimageds = augmentedImageDatastore(inputSize(1:2),images);
inx = 45;

layer = net.Layers(inx).Name;
features = activations(net,augimageds,layer,'OutputAs','rows');
deepnn.layerSamples = features;
deepnn.layerName = layer;
deepnn.networkName = dnnName;
deepnn.numStim = numImages;

dnn_ds.samples = features;
nstim = (idx)';
dnn_ds.sa.targets = nstim;

% compute the dissimilarity matrix
ds_dsm_dnn = cosmo_dissimilarity_matrix_measure(dnn_ds,'metric','correlation','center_data', true);
[samples, labels, values] = cosmo_unflatten(ds_dsm_dnn,1, 'set_missing_to',NaN);

deepnn.unflatten = samples;
deepnn.rdm = ds_dsm_dnn.samples;


name_file = fullfile([results_path, dnnName,'_features'] );
save (name_file, 'deepnn');

end