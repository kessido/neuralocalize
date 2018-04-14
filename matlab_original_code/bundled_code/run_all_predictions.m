%% Run piece-wise GLMs and do LOO predictions
%
% S.Jbabdi 04/2016

% %%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace the below your own
datadir='/vols/Scratch/HCP/rfMRI/subjectsD'; 
outdir='/path/to/results';
outdir='/vols/Scratch/saad/TMP_results';
unix(['mkdir -p ' outdir]);
addpath('./extras','./extras/CIFTIMatlabReaderWriter');
% %%%%%%%%%%%%%%%%%%%%%%%%%%
subjects=textread('./extras/subjects.txt','%s');


features_path = [outdir '/Features'];
task_path     = '/path/to/task/contrasts';    % the task contrasts should be CIFTI files as described below
task_path     = '/vols/Scratch/saad/MVPA_Functional_Localisation/Tasks';

[cifti,BM]    = open_wbfile('./extras/CIFTIMatlabReaderWriter/example.dtseries.nii');

ctx = [BM{1}.DataIndices(:);BM{2}.DataIndices(:)];
subctx = setdiff((1:91282)',ctx);

% Load all Features from all subjects (needs RAM)
disp('Load All Features');
AllFeatures=[];
for i=1:length(subjects)
    disp(['      ' subjects{i}]);
    % functional connectivity
    fname=[features_path '/' subjects{i} '_RFMRI_nosmoothing.dtseries.nii'];
    cifti=open_wbfile(fname);
    fc_features=cifti.cdata;
    % structural features
    fname=[features_path '/' subjects{i} '_Struct.dtseries.nii'];
    cifti=open_wbfile(fname);
    str_features=double(cifti.cdata);
    features = double([fc_features str_features]);    
    AllFeatures = cat(3,AllFeatures,features);
end
AllFeatures = permute(AllFeatures,[1 3 2]);
% normalise features
AllFeatures(ctx,:,:) = normalise(AllFeatures(ctx,:,:));
AllFeatures(subctx,:,:) = normalise(AllFeatures(subctx,:,:));


%% load all tasks
% Task contracts are CIFTI files
% One file per contrast, with all subjects concatenated in each file
nsubjects  = length(subjects);
ncontrasts = 86; % these contain both pos and neg contrast maps. There are 47 independent contrasts in there
AllTasks=zeros(91282,nsubjects,ncontrasts);
for i=1:86
    disp(i)
    cifti=open_wbfile([task_path '/AllSubjects_' zeropad(i,3) '.dtseries.nii']); 
    AllTasks(:,:,i) = cifti.cdata;    
end

%% Load spatial filters
% then threshold and do a winner-take-all
disp('Load Filters');
filters = open_wbfile([outdir '/ica_both_lowdim.dtseries.nii']);
[m,wta]=max(filters.cdata,[],2);
wta = wta .* (m>2.1);
S = zeros(size(filters.cdata));
for i=1:size(filters.cdata,2)
    S(:,i) = double(wta==i);
end

%% Run training and LOO predictions
disp('Start');
unix(['mkdir -p ' outdir '/Betas'])
unix(['mkdir -p ' outdir '/Predictions'])

for contrastNum = 1:86
    disp(['contrast ' zeropad(contrastNum,3)]);
    
    % start learning
    disp('--> start Learning');
    featIdx=1:size(AllFeatures,3);
    
    for i=1:length(subjects)
        subj=subjects{i};
        disp(['      ' subj]);
        task=AllTasks(:,i,contrastNum);
        Features = [ones(size(task,1),1) normalise(squeeze(AllFeatures(:,i,featIdx)))]; 
        % do the GLM
        betas=zeros(size(Features,2),size(S,2));
        for j=1:size(S,2)
            ind = S(:,j)>0;
            y = task(ind); M=[Features(ind,1) demean(Features(ind,2:end))];
            betas(:,j) = pinv(M)*y;
        end
        save([outdir '/Betas/contrast_' zeropad(contrastNum,3) '_' subj '_betas.mat'],'betas');        
    end
    
    % leave-one-out betas
    disp('--> LOO');
    for i=1:length(subjects)     
        subj=subjects{i};
        disp(['      ' subj]);
        loo_betas=zeros(size(Features,2),size(S,2),length(subjects)-1);
        cnt=1;
        for j=setdiff(1:length(subjects),i)
            load([outdir '/Betas/contrast_' zeropad(contrastNum,3) '_' subjects{j} '_betas.mat'],'betas');
            loo_betas(:,:,cnt) = betas;
            cnt=cnt+1;
        end
        save([outdir '/Betas/contrast_' zeropad(contrastNum,3) '_' subjects{i} '_loo_betas.mat'],'loo_betas');
    end
    
    disp('--> Predict');
    % predict
    X=zeros(91282,98);
    for i=1:length(subjects);     
        subj=subjects{i};
        disp(['      ' subj]);
        task=AllTasks(:,i);        
        Features = [ones(size(task,1),1) normalise(squeeze(AllFeatures(:,i,featIdx)))];
        load([outdir '/Betas/contrast_' zeropad(contrastNum,3) '_' subj '_loo_betas.mat'],'loo_betas');
        
        pred=0*task;
        for j=1:size(S,2)
            ind = S(:,j)>0;
            M=[Features(ind,1) demean(Features(ind,2:end))];
            pred(ind) = M*mean(loo_betas(:,j,:),3);
        end
        X(:,i)=pred;
        
    end
    cifti.cdata=X;
    ciftisave(cifti,[outdir '/Predictions/contrast_' zeropad(contrastNum,3) '_pred_loo.dtseries.nii']);
    
    
end

