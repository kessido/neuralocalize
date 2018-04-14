%% Structural Feature extraction
%  Extract the following structural features:
%   Cutical curvature
%   Sulcal depth
%   Cortical thickness
%   Myelin Map
%   Diffusion features
%
% S.Jbabdi 04/2016

% %%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace the below your own
datadir='/vols/Scratch/HCP/Structural/Q123'; 
outdir='/path/to/results';
outdir='/vols/Scratch/saad/TMP_results';
addpath('./extras/CIFTIMatlabReaderWriter');
% %%%%%%%%%%%%%%%%%%%%%%%%%%

subjects=textread('./extras/subjects.txt','%s');

% Load subject's data
for s=1:length(subjects)
    subj=subjects{s};
    disp(subj);
    x=zeros(91282,3); F=[];
    surfdir=[datadir '/' subj '/MNINonLinear/fsaverage_LR32k'];
    
    fname=[surfdir '/' subj '.L.curvature.32k_fs_LR.shape.gii'];
    
    % Curvature
    L=open_wbfile([surfdir '/' subj '.L.curvature.32k_fs_LR.shape.gii']);
    R=open_wbfile([surfdir '/' subj '.R.curvature.32k_fs_LR.shape.gii']);
    x=x(:,1)*0;
    x(BM{1}.DataIndices) = L.cdata(BM{1}.SurfaceIndices);
    x(BM{2}.DataIndices) = R.cdata(BM{2}.SurfaceIndices);    
    F=[F x];

    % Sulcal depth
    L=open_wbfile([surfdir '/' subj '.L.sulc.32k_fs_LR.shape.gii']);
    R=open_wbfile([surfdir '/' subj '.R.sulc.32k_fs_LR.shape.gii']);
    x=x(:,1)*0;
    x(BM{1}.DataIndices) = L.cdata(BM{1}.SurfaceIndices);
    x(BM{2}.DataIndices) = R.cdata(BM{2}.SurfaceIndices);    
    F=[F x];

    % Cortical thickness
    L=open_wbfile([surfdir '/' subj '.L.thickness.32k_fs_LR.shape.gii']);
    R=open_wbfile([surfdir '/' subj '.R.thickness.32k_fs_LR.shape.gii']);
    x=x(:,1)*0;
    x(BM{1}.DataIndices) = L.cdata(BM{1}.SurfaceIndices);
    x(BM{2}.DataIndices) = R.cdata(BM{2}.SurfaceIndices);    
    F=[F x];
        
    % Myelin map (T1/T2)
    L=open_wbfile([surfdir '/' subj '.L.SmoothedMyelinMap.32k_fs_LR.func.gii']);
    R=open_wbfile([surfdir '/' subj '.R.SmoothedMyelinMap.32k_fs_LR.func.gii']);
    x=x(:,1)*0;
    x(BM{1}.DataIndices) = L.cdata(BM{1}.SurfaceIndices);
    x(BM{2}.DataIndices) = R.cdata(BM{2}.SurfaceIndices);    
    F=[F x];
    
    % Some features from diffusion
    x = dmri_get_features(subj);
    F=[F x];

    dt=cifti; dt.cdata=F;
    ciftisave(dt,[outdir '/Features/' subj '_Struct.dtseries.nii']);

end





