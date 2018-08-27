# %% Group Incremental PCA (Smith et al. 2014, PMC4289914)
# %  Run PCA on random list of 100 subjects (subset of HCP)
#
# % S.Jbabdi 04/2016
# Neuro-sience sementar python version.

import utils.folders
import utils.cifti_utils
import os
import numpy as np
import sklearn.preprocessing

def demean(np_array):
    np_array - np.mean(np_array, keepdims=True)

def variance_normalise(np_array):
    old_mean = np.mean(np_array)
    np_array = sklearn.preprocessing.normalize(np_array)
    reap_mean = np.repeat(old_mean, (1,np_array.shape[1])) 
    return np_array + reap_mean

def func(
        datadir='/vols/Scratch/HCP/rfMRI/subjectsD',
        outdir='/path/to/results',
        subjects=['100307'],
        # Keep components
        dPCAint=1200, dPCA=1000,
        outname='rand100'):
    utils.folders.create_dir(outdir)
    sessions = [('1' 'LR'), ('1' 'RL'), ('2' 'LR'), ('2' 'RL')]  # pretefy

    # Loop over sessions and subjects
    W = []
    for a, b in sessions:
        for subject in subjects:
            print(subject)
            subjdir = os.path.join(datadir, subject, 'MNINonLinear/Results/')
            fname = os.path.join(
                subjdir, f'/rfMRI_REST{a}_{b}/rfMRI_REST{a}_{b}_Atlas_hp2000_clean.dtseries.nii')

            # % read and demean data
            print('read data')
            cifti, BM = utils.cifti_utils.load_nii_brain_data_from_file(fname)
            cifti = np.asarray(cifti, dtype=np.float32)
            grot = demean(cifti)

            # % noise variance normalisation
            grot = variance_normalise(grot)
            # % concat
            W = np.concatenate(W, demean(grot))
            # % PCA reduce W to dPCAint eigenvectors
            print(f'do PCA {num2str(size(W,1))} {x} {num2str(size(W,2))}')
            uu, dd = eigs(W@W.transpose(), min(dPCAint, size(W, 1)-1))
            W = uu.transpose()@W
    data = W[1: dPCA, :].transpose()

    # % Save group PCA results
    # dt = utils.cifti_utils.load_nii_brain_image_from_file('./extras/CIFTIMatlabReaderWriter/example.dtseries.nii')
    # dt = data
    result_path = os.path.join(
        outdir, f'/GROUP_PCA_{outname}_RFMRI.dtseries.nii')
    utils.cifti_utils.save_image_to_file(data, result_path)
