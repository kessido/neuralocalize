function BM = load_BM()
% BM = load_BM()

addpath ~/matlab/CIFTIMatlabReaderWriter_old
[~,BM]=open_wbfile('~saad/data/example_gifti_cifti/example.dscalar.nii');