%% Group Incremental PCA (Smith et al. 2014, PMC4289914)
%  Run PCA on random list of 100 subjects (subset of HCP)
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


subjects = textread('./extras/subjects_rand100.txt','%s');
sessions = {'1' 'LR';'1' 'RL';'2' 'LR';'2' 'RL'};

% Keep components
dPCAint=1200;   
dPCA=1000;


% Loop over sessions and subjects
W= [] ;
for sess = 1:4
    a=sessions{sess,1};b=sessions{sess,2};
       
    for s=1:length(subjects)
        subj=subjects{s};
        disp(subj);
        subjdir=[datadir '/' subj '/MNINonLinear/Results/' ];
        fname=[subjdir '/rfMRI_REST' a '_' b '/rfMRI_REST' a '_' b '_Atlas_hp2000_clean.dtseries.nii'];
        
        % read and demean data
        disp('read data');
        [cifti,BM]=open_wbfile(deblank(fname));        
        grot=demean(double(cifti.cdata)'); clear cifti.cdata;        
        
        % noise variance normalisation
        grot = variance_normalise(grot);
        % concat 
        W=[W; demean(grot)]; clear grot;
        % PCA reduce W to dPCAint eigenvectors
        disp(['do PCA ' num2str(size(W,1)) 'x' num2str(size(W,2))]);
        [uu,dd]=eigs(W*W',min(dPCAint,size(W,1)-1));  W=uu'*W; clear uu;                            
    end
    
end
data=W(1:dPCA,:)';

% Save group PCA results
dt=open_wbfile('./extras/CIFTIMatlabReaderWriter/example.dtseries.nii');
dt.cdata=data;
ciftisave(dt,[outdir '/GROUP_PCA_rand100_RFMRI.dtseries.nii']);







