%% MVPA - LOOKALIKE -- ANIMALS VS OBJECTS

clear all;
close all;

% add path
addpath(genpath('C:\Users\psiok\Desktop\Spring_2022\fMRI_HandsOn\CoSMoMVPA-master'));

ROInames = {'V1' 'VTC_post' 'VTC_ant'};

for iSub = 1:12
    for iROI = 1:3
        
        ROI = ROInames{iROI};
        
        % Define data
        glm_fn = sprintf('C:/Users/psiok/Desktop/Spring_2022/fMRI_HandsOn/FinalAss/fmri_data/SUB%02d/SPM.mat',iSub);
        
        % Load the dataset with mask
        msk_fn = sprintf('C:/Users/psiok/Desktop/Spring_2022/fMRI_HandsOn/FinalAss/ROIs/%s.nii',ROI);
        
        ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);
        
        % simple sanity check to ensure all attributes are set properly
        cosmo_check_dataset(ds);
        
        % remove constant features
        ds = cosmo_remove_useless_data(ds);
        
        % define targets and category dimensions
        nConditions = 27;
        nRuns = 12;
        ds.sa.targets = repmat((1:9)',36,1);
        ds.sa.dim = reshape(repmat((1:3),9,12),[],1);
        
        for iTest = 1:2
            
            if iTest == 1 % Lookalike vs Objects
                
                idx = cosmo_match(ds.sa.dim,[1 3]); 
                ds_sel = cosmo_slice(ds,idx);       
                
            elseif iTest == 2 % lookalike vs Animals
                
                idx =cosmo_match(ds.sa.dim,[1 2]);
                ds_sel = cosmo_slice(ds,idx);  
            end
            
            %% Define classifier
            args.classifier = @cosmo_classify_lda;
            args.normalization = 'demean';
            
            %% Define partitions
            args.partitions = cosmo_nchoosek_partitioner(ds_sel, 1, 'dim', []);
            
            %% decode using the measure (cosmo_crossvalidate)
            ds_accuracy = cosmo_crossvalidation_measure(ds_sel, args);
            fprintf('Test %d, Sub %d, %s, accuracy: %.3f\n', iTest, iSub, ROI, ds_accuracy.samples);
            
            allRes(iSub, iROI, iTest) = ds_accuracy.samples;
            
        end
    end
end

meanAcc = mean(allRes); % mean
semAcc = std(allRes)/sqrt(12); % std err of mean

TestNames = {'Lookalike vs Objects','Lookalike vs Animals'};
Regions = ({'V1','VTC post','VTC ant'});

for iTest = 1:2
    
    subplot(1,2,iTest);
    
    bar(meanAcc(1,:,iTest));
    hold on;
    errorbar(meanAcc(1,:,iTest),semAcc(1,:,iTest),'.');
    ylabel('accuracy');
    ylim([0 0.25]);
    line([0 length(Regions)+1],[0.11 0.11]); % the line indicating accuracy at chance (since we have 9 stimuli I set chance level at 0.11)
    
    set(gca, 'XTick', 1:length(Regions), 'XTickLabel', Regions); % labels
    title(TestNames{iTest});
end

[H P CI T]=ttest(allRes,0.11,0.05,'right') % test for significance
