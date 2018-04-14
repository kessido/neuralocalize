function F = dmri_get_features(subj)
% F = dmri_extract_features(subj)
%
% Extract the following features from diffusion MRI:
%   1. Mean diffusivity (MD) calculated in 3 different b-values
%   2. Fractional Anisotropy (FA)
%   3. The dot-product between the principal diffusion direction and the
%      cortical surface at the interface between grey and white matter
%      (N*V1)
%
% ------------------------------------------------------------------------

addpath('./extras');
addpath([getenv('FSLDIR') '/etc/matlab']);
addpath('/home/fs0/saad/matlab/toolbox_graph'); % from https://github.com/gpeyre

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace with your own
diffdir   = '/vols/Scratch/HCP/Diffusion/Q123';
structdir = '/vols/Scratch/HCP/Structural/Q123';
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get header info for output
[~,bm]=open_wbfile('./extras/CIFTIMatlabReaderWriter/example.dtseries.nii');



% Features: FA MD1 MD2 MD3 N*V1
% Assumes that diffusion tensor has been pre-calculated on each shell
% and stored in a file called
% /diffdir/subjectID/T1w/Diffusion/dtifit_b{1,2,3}k/dti_tensor.nii.gz

% Project tensor onto cortical surface and open into matlab
[~,tmpname]=unix([getenv('FSLDIR'),'/bin/tmpnam']);
tmpname=deblank(tmpname);
tensor=cell(3,1);
for b=1:3
    tensnifti = [diffdir '/' subj '/T1w/Diffusion/dtifit_b' num2str(b) 'k/dti_tensor.nii.gz'];
    tenscifti = [tmpname '_dti_tensor.dtseries.nii'];
    
    surfL=[structdir '/' subj '/T1w/fsaverage_LR32k/' subj '.L.midthickness.32k_fs_LR.surf.gii'];
    surfR=[structdir '/' subj '/T1w/fsaverage_LR32k/' subj '.R.midthickness.32k_fs_LR.surf.gii'];
    
    unix(['wb_command -volume-to-surface-mapping ' tensnifti ' ' surfL ' ' tmpname '_L.func.gii -trilinear ']);
    unix(['wb_command -volume-to-surface-mapping ' tensnifti ' ' surfR ' ' tmpname '_R.func.gii -trilinear ']);
    unix(['wb_command -cifti-create-dense-timeseries ' tenscifti ' -left-metric ' tmpname '_L.func.gii -right-metric ' tmpname '_R.func.gii']);
    
    [tensor{b},BM]=open_wbfile(tenscifti);
end

% Loop over 3 shells and calculate tensor scalars
N = size(tensor{1}.cdata,1);
vals={'FA' 'MD' 'V1'};
tensVals=cell(3,length(vals));
for b=1:3
    % Create a pseudo-volume to perform tensor decomposition on using FSL
    T = zeros(N,6);
    T(1:N,:) = tensor{b}.cdata;
    T = reshape(T,[8123,8,1,6]);
    
    [~,tmpname]=unix('tmpnam');
    tmpname=deblank(tmpname);
    
    save_avw(T,tmpname,'f',[1 1 1]);
    % Perform tensor decomposition
    unix(['fslmaths ' tmpname ' -tensor_decomp ' tmpname]);
    % Save the tensor scalars as surfaces
    for i=1:length(vals)
        x=squeeze(read_avw([tmpname '_' vals{i}]));
        x=reshape(x,N,size(x,3));
        tensVals{b,i}=zeros(91282,size(x,2));
        tensVals{b,i}(bm{1}.DataIndices,:) = x(BM{1}.DataIndices(bm{1}.SurfaceIndices),:);
        tensVals{b,i}(bm{2}.DataIndices,:) = x(BM{2}.DataIndices(bm{2}.SurfaceIndices),:);
    end
end

% Compute surface normals (for the N*V1 feature)
hemis={'L' 'R'};
n=cell(2,1);
for h=1:2
    surf=[structdir '/' subj '/T1w/fsaverage_LR32k/' subj '.' hemis{h} '.white.32k_fs_LR.surf.gii'];
    gii=open_wbfile(surf);
    n{h}=compute_normal(gii.vertices,gii.faces)';
end

N=zeros(91282,3);
N(bm{1}.DataIndices,:) = n{1}(bm{1}.SurfaceIndices,:);
N(bm{2}.DataIndices,:) = n{2}(bm{2}.SurfaceIndices,:);

F=tensVals{1,1};                                        % FA
F=[F 1e3*[tensVals{1,2} tensVals{2,2} tensVals{3,2}]];  % convert MD to um^2/ms
F=[F abs(sum(N.*tensVals{1,3},2))];                     % abs(N*V1)

F(isnan(F))=0;F(isinf(F))=0;

