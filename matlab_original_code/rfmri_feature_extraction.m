%% RFMRI Feature extraction
%  ICA + other tricks:
%   a. Run group ICA on group_pca output
%   b. Perform dual regression to produce subject-specific cortical ICA
%      maps
%   c. Obtain subcortical parcellation
%   d. Produce individual subject's connectivity maps
%
% S.Jbabdi 04/2016

% %%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace the below your own
datadir='/vols/Scratch/HCP/rfMRI/subjectsD'; 
outdir='/path/to/results';
unix(['mkdir -p ' outdir]);
addpath('./extras','./extras/CIFTIMatlabReaderWriter');
addpath('/path/to/FastICA_25');  % From http://research.ics.aalto.fi/ica/fastica
% %%%%%%%%%%%%%%%%%%%%%%%%%%
subjects=textread('./extras/subjects.txt','%s');
sessions={'1' 'LR';'1' 'RL';'2' 'LR';'2' 'RL'};

% Read group PCA results and split Left/Right hemispheres
[cifti,BM]=open_wbfile('./extras/GROUP_PCA_200_RFMRI.dtseries.nii');

LH = cifti_extract_data(cifti,'L',BM);
RH = cifti_extract_data(cifti,'R',BM);


%% A - Run group ICA for each hemisphere separately

numIC  = 40;
ica_LH = fastica(LH','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);
numIC  = 40;
ica_RH = fastica(RH','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);

% flip signs (large tail on the right)
ica_LH = ica_LH .* repmat(sign(sum(sign(ica_LH.*(abs(ica_LH)>2)),2)),1,size(ica_LH,2));
ica_RH = ica_RH .* repmat(sign(sum(sign(ica_RH.*(abs(ica_RH)>2)),2)),1,size(ica_RH,2));

% Keep ICA components that have L/R symmetry
% left-right DICE of cortical ICs to  
% 1) re-order the ICs
% 2) select the ICs that are found in both hemispheres

N=91282;
x=zeros(32492,size(ica_LH,1)); % Left
y=zeros(32492,size(ica_RH,1)); % Right

x(BM{1}.SurfaceIndices,:) = ica_LH';
y(BM{2}.SurfaceIndices,:) = ica_RH';
thr=2;
D     = dice( x>thr, y>thr );

Dthr  = (D == repmat(max(D,[],2),1,size(D,2)));
Dtmp  = ((D.*Dthr) == repmat(max(D.*Dthr,[],1),size(D,1),1));
Dthr  = Dtmp.*Dthr;

r     = find(sum(Dthr,2));
[~,c] = max(Dthr,[],2);
c     = c(r);
% save
x=zeros(N,length(r));
x(BM{1}.DataIndices,:)=ica_LH(r,:)';
x(BM{2}.DataIndices,:)=ica_RH(c,:)';
dt=cifti; dt.cdata=x;
ciftisave(dt,[outdir '/ica_LR_MATCHED.dtseries.nii']);

%% Run group ICA on both hemispheres (for use as spatial filters)
Both   = double(cifti.cdata);
numIC  = 50;
ica_both = fastica(Both','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);

% flip sign
ica_both = ica_both .* repmat(sign(sum(sign(ica_both.*(abs(ica_both)>2)),2)),1,size(ica_both,2));

% save
dt=cifti;
dt.cdata=ica_both';
ciftisave(dt,[outdir '/ica_both_lowdim.dtseries.nii']);



%% B - Dual regression 
dt=open_wbfile([outdir '/ica_LR_MATCHED.dtseries.nii']);
N_LH = size(dt.cdata,2);
N_RH = size(dt.cdata,2);

G = zeros(91282,N_LH+N_RH);
G(BM{1}.DataIndices,1:N_LH)           = dt.cdata(BM{1}.DataIndices,:);
G(BM{2}.DataIndices,N_LH+1:N_LH+N_RH) = dt.cdata(BM{2}.DataIndices,:);


Hemis = zeros(91282,N_LH+N_RH);
Hemis(BM{1}.DataIndices,1:N_LH)           = 1;
Hemis(BM{2}.DataIndices,N_LH+1:N_LH+N_RH) = 1;

pinvG = pinv(G);

unix(['mkdir -p ' outdir '/DR']);
for s=1:length(subjects)
    subj=subjects{s};
    disp(subj);
    data = [];
    disp(' read and demean data');
    for sess = 1:4
        disp(['session ' num2str(sess)]);
        a=sessions{sess,1};b=sessions{sess,2};        
        subjdir=[datadir '/' subj '/MNINonLinear/Results' ];
        fname=[subjdir '/rfMRI_REST' a '_' b '/rfMRI_REST' a '_' b '_Atlas_hp2000_clean.dtseries.nii'];
        cifti=open_wbfile(deblank(fname));
        cifti.cdata = variance_normalise(double(cifti.cdata)')';
        data=[data detrend(double(cifti.cdata)')'];
    end
    oname=[outdir '/DR/' subj '_DR2_nosmoothing.dtseries.nii'];
    % DR - Step 1 (get individual time series)
    disp('DR - step 1');
    T = pinvG*data;
    % DR - Step 2 (get individual spatial maps)
    disp('DR - step 2');
    [cope,varcope,stats] = fsl_glm(T',data');
    % saving subjectwise maps
    cifti.cdata = stats.t' .* Hemis;
    ciftisave(cifti,oname);
end


%% C - Get subcortical parcellation
% subcortex
SC_labels     = cell(21,1);
SC_labels{1}  = 'CIFTI_STRUCTURE_CORTEX_LEFT';
SC_labels{2}  = 'CIFTI_STRUCTURE_CORTEX_RIGHT';
SC_labels{3}  = 'CIFTI_STRUCTURE_ACCUMBENS_LEFT';
SC_labels{4}  = 'CIFTI_STRUCTURE_ACCUMBENS_RIGHT';
SC_labels{5}  = 'CIFTI_STRUCTURE_AMYGDALA_LEFT';
SC_labels{6}  = 'CIFTI_STRUCTURE_AMYGDALA_RIGHT';
SC_labels{7}  = 'CIFTI_STRUCTURE_BRAIN_STEM';
SC_labels{8}  = 'CIFTI_STRUCTURE_CAUDATE_LEFT';
SC_labels{9}  = 'CIFTI_STRUCTURE_CAUDATE_RIGHT';
SC_labels{10} = 'CIFTI_STRUCTURE_CEREBELLUM_LEFT';
SC_labels{11} = 'CIFTI_STRUCTURE_CEREBELLUM_RIGHT';
SC_labels{12} = 'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT';
SC_labels{13} = 'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT';
SC_labels{14} = 'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT';
SC_labels{15} = 'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT';
SC_labels{16} = 'CIFTI_STRUCTURE_PALLIDUM_LEFT';
SC_labels{17} = 'CIFTI_STRUCTURE_PALLIDUM_RIGHT';
SC_labels{18} = 'CIFTI_STRUCTURE_PUTAMEN_LEFT';
SC_labels{19} = 'CIFTI_STRUCTURE_PUTAMEN_RIGHT';
SC_labels{20} = 'CIFTI_STRUCTURE_THALAMUS_LEFT';
SC_labels{21} = 'CIFTI_STRUCTURE_THALAMUS_RIGHT';

% Accumbens    = 1 cluster per hemisphere
% Amygdala     = 2 clusters per hemisphere
% Brainstem    = No
% Caudate      = 2 clusters per hemisphere
% Cb           = ICA (6 per hemisphere)
% Diencephalon = No
% Hippocampus  = 2 clusters (reord2) per hemisphere
% Pallidum     = 1 cluster per hemisphere
% Putamen      = 2 clusters per hemisphere
% Thalamus     = ICA (4 per hemisphere)
[cifti,BM]=open_wbfile('./extras/GROUP_PCA_200_RFMRI.dtseries.nii');


SC_clusters =[];SC_structures=[];
x=zeros(size(cifti.cdata,1),1);
% Accumbens (2)
x=0*x;x(BM{3}.DataIndices) = 1; 
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*3]; 
x=0*x;x(BM{4}.DataIndices) = 1; 
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*4]; 
% Amygdala (2)
Y = cifti_extract_data(cifti,SC_labels{5},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{5}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*5]; 
Y = cifti_extract_data(cifti,SC_labels{6},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{6}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*6]; 
% Caudate (4)
Y = cifti_extract_data(cifti,SC_labels{8},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{8}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*8]; 
Y = cifti_extract_data(cifti,SC_labels{9},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{9}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*9]; 
% Cb (6)
Cb = cifti_extract_data(cifti,SC_labels{10},BM);
numIC  = 3;
ica_Cb = fastica(Cb','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);
ica_Cb = ica_Cb .* repmat(sign(sum(sign(ica_Cb.*(abs(ica_Cb)>2)),2)),1,size(ica_Cb,2));
x=zeros(size(cifti.cdata,1),size(ica_Cb,1));
x(BM{10}.DataIndices,:) = ica_Cb';
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*10]; 
Cb = cifti_extract_data(cifti,SC_labels{11},BM);
numIC  = 3;
ica_Cb = fastica(Cb','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);
ica_Cb = ica_Cb .* repmat(sign(sum(sign(ica_Cb.*(abs(ica_Cb)>2)),2)),1,size(ica_Cb,2));
x=zeros(size(cifti.cdata,1),size(ica_Cb,1));
x(BM{11}.DataIndices,:) = ica_Cb';
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*11]; 
% Hippocampus (4)
Y = cifti_extract_data(cifti,SC_labels{14},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{14}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*14]; 
Y = cifti_extract_data(cifti,SC_labels{15},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{15}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*15]; 
% Pallidum (2)
x=0*x(:,1);x(BM{16}.DataIndices) = 1; 
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*16]; 
x=0*x(:,1);x(BM{17}.DataIndices) = 1; 
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*17]; 
% Putamen (4)
Y = cifti_extract_data(cifti,SC_labels{18},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{18}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*18]; 
Y = cifti_extract_data(cifti,SC_labels{19},BM);
p = reord2(1+corrcoef(Y'),0,1);
x=0*[x(:,1) x(:,1)];x(BM{19}.DataIndices,:) = double([p>0 p<0]);
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*19]; 
% Thalamus (6)
Th = cifti_extract_data(cifti,SC_labels{20},BM);
numIC  = 3;
ica_Th = fastica(Th','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);
ica_Th = ica_Th .* repmat(sign(sum(sign(ica_Th.*(abs(ica_Th)>2)),2)),1,size(ica_Th,2));
x=zeros(size(cifti.cdata,1),size(ica_Th,1));
x(BM{20}.DataIndices,:) = ica_Th';
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*20]; 
Th = cifti_extract_data(cifti,SC_labels{21},BM);
numIC  = 3;
ica_Th = fastica(Th','approach', 'symm', 'g', 'tanh','lastEig',numIC,'numOfIC', numIC);
ica_Th = ica_Th .* repmat(sign(sum(sign(ica_Th.*(abs(ica_Th)>2)),2)),1,size(ica_Th,2));
x=zeros(size(cifti.cdata,1),size(ica_Th,1));
x(BM{21}.DataIndices,:) = ica_Th';
SC_clusters = [SC_clusters x]; SC_structures = [SC_structures ones(1,size(x,2))*21]; 


% save results
dt=cifti; dt.cdata=SC_clusters;
ciftisave(dt,[outdir '/SC_clusters.dtseries.nii']);




%% D - Final feature extraction (forming semi-dense connectome)
% For each subject, load RFMRI data
% then load ROIs from above
% calculate semi-dense connectome

SC=open_wbfile([outdir '/SC_clusters.dtseries.nii']);SC=double(SC.cdata);

unix(['mkdir -p ' outdir '/Features']);
for s=1:length(subjects)
    subj=subjects{s};
    disp(subj);
    oname=[outdir '/DR/' subj '_DR2_nosmoothing.dtseries.nii'];
    LHRH=open_wbfile(oname);LHRH=double(LHRH.cdata);
    ROIS=[LHRH SC];
    % load RFMRI data
    W=[];
    for sess = 1:4
        a=sessions{sess,1};b=sessions{sess,2};    
        subjdir=[datadir '/' subj '/MNINonLinear/Results' ];
        fname=[subjdir '/rfMRI_REST' a '_' b '/rfMRI_REST' a '_' b '_Atlas_hp2000_clean.dtseries.nii'];
        % read and demean data
        disp('read data');
        [cifti,BM]=open_wbfile(deblank(fname));
        grot=demean(double(cifti.cdata)'); clear cifti.cdata;
        % noise variance normalisation
        grot = variance_normalise(grot);
        % concat 
        W=[W; demean(grot)]; clear grot;
    end
    % Feature extraction
    % 1. multiple regression
    T = pinv(ROIS)*W';
    % 2. correlation coefficient
    F = (normalise(T,2)*normalise(W,1))';
    dt=cifti; dt.cdata=F;
    ciftisave(dt,[outdir '/Features/' subj '_RFMRI_nosmoothing.dtseries.nii']);    
    
end
