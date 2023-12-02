%% 1.RSA: NEURAL DATA(aVTC) & MODELS

% STEP 1: GENERATE DS

clear all
config = cosmo_config();

addpath ('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\CoSMoMVPA-master');

% reset citation list
cosmo_check_external('-tic');

% Load data
subjs =  {'SUB01','SUB02','SUB03','SUB04','SUB05','SUB06','SUB07','SUB08','SUB09','SUB10','SUB11','SUB12'};
numSubjs = size (subjs, 2);

naROI =  {'VTC_ant'};
nameROIs = size (naROI, 2);

ROI =  'VTC_ant.nii';

%set paths
study_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\fmri_data\'); %fmri data
mask_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\ROIs\'); % ROI
results_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\DS\');

for s = 1:numSubjs
    
    data_path = fullfile([study_path, subjs{s}]); %sub
    mask_fn = fullfile(mask_path, ROI); % ROI
    
    data_fn=fullfile(data_path, 'SPM.mat');
    
    % define targets and chunks
    nConditions = 27;
    nRuns = 12;
    targets = repmat((1:nConditions)',nRuns,1);
    chunks = reshape(repmat((1:nRuns),nConditions,1),[],1);
    
    % generate COSMO DS structure
    ds = cosmo_fmri_dataset(data_fn,'mask', mask_fn, 'targets', targets, 'chunks',chunks);
    
    % simple sanity check to ensure all attributes are set properly
    cosmo_check_dataset(ds);
    
    % remove constant features
    ds = cosmo_remove_useless_data(ds); % remove non-values
    
    % save ds
    name_file = fullfile([results_path, subjs{s}, '_VTC_ant_ds']);
    save (name_file, 'ds');
    
end

% STEP 2: COMPUTE RDM

%set paths
study_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\');

for r = 1:numSubjs
    
    %Load fMRI DS
    data_file = strcat(subjs(r),'_VTC_ant_ds.mat'); % constructing file name
    ds_filepath = fullfile(study_path, 'DS'); % stating ds pathway
    
    ds = fullfile(ds_filepath, string(data_file)); % finding ds
    load (ds); % loading ds
    
    % simple sanity check to ensure all attributes are set properly
    cosmo_check_dataset(ds);
    
    %remove constant features
    ds = cosmo_remove_useless_data(ds);
    
    % compute avg across runs with cosmo cosmo_fx
    f_ds = cosmo_fx(ds, @(x)mean(x,1), 'targets'); % take the average with this function
    
    % compute RDM with cosmo_dissimilarity_matrix_measure
    ds_dsm = cosmo_dissimilarity_matrix_measure(f_ds,'center_data',true); % correlation is the default mode --Alternatively ('metric', 'correlation', 'center data', true)
    
    %  visualize RDM
    [samples, labels, values] = cosmo_unflatten(ds_dsm,1,'set_missing_to',NaN);
    %imagesc(samples)
    
    %store results dsm (vector);
    temp_dsm = ds_dsm.samples;
    dsm_all(:,r) = temp_dsm;
    
    %store results matrix;
    temp_dsm_mat = samples;
    dsm_mat_all(:,:,r)= temp_dsm_mat;
    
end

dsm_vect = dsm_all;
dsm_unflatten = dsm_mat_all;

% store results
RDM.data = dsm_vect;
RDM.data_unflatten = dsm_unflatten;
RDM.ROIs = 'VTC_ant';
RDM.SUB = subjs;

results_path=fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\');

name_file=fullfile([results_path, 'SingleSubj_ROIs_RDM']);
save (name_file, 'RDM');

% STEP 3: RSA BETWEEN NEURAL DATA AND MODELS

data_filepath = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\');

% Load models
model_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\models\');
load([model_path, 'mod.mat']);
mod = models;

% Load neural data
RDM_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\');
load ([RDM_path, 'SingleSubj_ROIs_RDM']);
neural_vect = RDM.data;

% Running RSA with partialcorri and compare the results
tempRSA_parCorr = partialcorri(neural_vect, mod.vec_dissimilarity,'Type','Pearson');
RSA_parCorr = tempRSA_parCorr;

% fisher transformation the RSA results
fisher_tempRSA_parcorr = atanh(tempRSA_parCorr);
RSA_fisher_corr = fisher_tempRSA_parcorr;

[h,p,ci,stats] = ttest(fisher_tempRSA_parcorr);
stat.h = h;
stat.p = p;
stat.stats = stats;
stat.rois = RDM.ROIs;

% compute noise ceiling

for s = 1:size(neural_vect,2)
    singleSubj = neural_vect(:,s);
    maskMinus = neural_vect;
    maskMinus(:,s) = NaN;
    groupMinus = nanmean(maskMinus,2);
    acrossSubj_lower_bound(s,1) = corr(singleSubj, groupMinus);
end

stat.noiseCeilling = mean(acrossSubj_lower_bound(:,1));
statistics = stat;

% STEP 4: PLOTTING RESULTS WITH NOISE CEILING

corr_mean = mean(RSA_fisher_corr);
std_err = std(RSA_fisher_corr)/sqrt(12);

graph = bar(corr_mean);
title('RSA between neural data & models')
xticklabels({'appearance','animacy'})
xlabel('aVTC')
ylabel('Correlation Coefficient')
yline(0.454, '-.b','LineWidth',2)

hold on
errorbar(corr_mean,std_err,'.r');
hold off

%% 2. RSA: BEHAVIORAL RATINGS & MODELS

study_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\');

% Load models
model_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\models\');
load([model_path, 'mod.mat']);
mod = models;

% Load beahvioral data
behavidata_path = fullfile(study_path, '/behavioural_singlesubj.mat');
load(behavidata_path);
behav_singlesubjs = behavi_dissimilarity_singlesubj(:,1:12); % I took first 12 columns since we have 12 subjects
%behav_singlesubjs = behavi_dissimilarity_singlesubj;

% Compute RSA between behavioral data & models
RSA_parcorr_behav = partialcorri(behav_singlesubjs,mod.vec_dissimilarity,'Type','Pearson');
fisher_RSA_parcorr_behav = atanh(RSA_parcorr_behav);
RSA_fisher_corr_behav = fisher_RSA_parcorr_behav;

% Compute noise ceiling

for s = 1:size(behav_singlesubjs,2)
    singleSubj = behav_singlesubjs(:,s);
    maskMinus = behav_singlesubjs;
    maskMinus(:,s) = NaN;
    groupMinus = nanmean(maskMinus,2);
    acrossSubj_lower_bound1(s,1) = corr(singleSubj, groupMinus);
end

noiseCeilling_behav = mean(acrossSubj_lower_bound1(:,1));

% Plot results with noise ceiling

corr_mean_behav = mean(RSA_fisher_corr_behav);
std_err_behav = std(RSA_fisher_corr_behav)/sqrt(12);

graph = bar(corr_mean_behav);
title('RSA between behavioral data & models')
xticklabels({'appearance','animacy'})
xlabel('Models')
ylabel('Correlation Coefficient')
yline(0.4664, '-.b','LineWidth',2)

hold on
errorbar(corr_mean_behav,std_err_behav,'.r');
hold off

%% 3. RSA: DNN (vgg-19) & MODELS
% I extracted DNN features by using an additional function script. It is
% called 'dnn_features_vgg19' and I put inside folder that I upload.

dnnName = 'vgg19';
dnndata_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\results_DNN\'); % dnn feature

% Load models
model_path = fullfile('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\FinalAss\models\');
load([model_path, 'mod.mat']);
mod = models;

dnn_path = fullfile([dnndata_path, dnnName '_features']); 
load(dnn_path);
data_dnn = deepnn.rdm;

tempRSA_parcorr_dnn = partialcorri(data_dnn,mod.vec_dissimilarity,'Type','Pearson');
RSA_parcorr_dnn = tempRSA_parcorr_dnn; % store part corr results

%fisher transf neural data
fisher_tempRSA_parcorr_dnn = atanh(tempRSA_parcorr_dnn);
RSA_fisher_parcorr_dnn = fisher_tempRSA_parcorr_dnn;

RSA_fisher_dnn = RSA_fisher_parcorr_dnn;
RSA_raw_dnn = RSA_parcorr_dnn;

% visual inspection
graph = bar(RSA_fisher_dnn);
title('RSA between DNN data & models')
xticklabels({'appearance','animacy'})
xlabel('VGG-19 Layer fc8')
ylabel('Correlation Coefficient')




