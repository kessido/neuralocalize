# %% Group Incremental PCA (Smith et al. 2014, PMC4289914)
# %  Run PCA on random list of 100 subjects (subset of HCP)
#
# % S.Jbabdi 04/2016
# Neuro-sience sementar python version.

import os

import numpy as np
import sklearn.preprocessing
from constants import dtype
import utils.cifti_utils


#
# def demean(np_array):
#     return sklearn.preprocessing.scale(np_array, with_std=False)
#
#
# def ss_svds(x, n):
#     if x.shape[0] < x.shape[1]:
#         if n < x.shape[0]:
#             v, u = scipy.sparse.linalg.eigs(x @ x.transpose(), n)
#         else:
#             v, u = scipy.linalg.eig(x @ x.transpose())
#             u = np.fliplr(u)
#             v = np.fliplr(v)
#         s = np.diag(np.sqrt(np.abs(v)))
#         v = x.transpose() @ (u * np.diag((1. / np.diag(s))))
#     else:
#         if n < x.shape[1]:
#             d, v = scipy.sparse.linalg.eigs(x.transpose() @ x, n)
#         else:
#             d, v = np.linalg.eig(x.transpose() @ x)
#             d = np.flipud(np.fliplr(d))
#             v = np.fliplr(v)
#         s = np.sqrt(np.abs(d))
#         u = x @ (v @ np.diag((1. / np.diag(s))))
#     return u, s, v
#
#
# def variance_normalise(y):
#     yn = y
#     uu, ss, vv = ss_svds(y, 30)
#     vv[np.abs(vv) < 2.3 @ np.std(vv)] = 0
#     stddevs = np.max(np.std(yn - uu @ ss @ vv.transpose()), 0.001)
#     yn = yn / stddevs
#     return yn

def concatenate_matrices(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return np.concatenate((a, b))


# TODO(kess) check if the whiten on the PCA make it better.
# TODO(kess) use sklearn incremental PCA instead.
def iterative_pca_subject(current_component,
                          subject_image,
                          iterative_n):
    # In Ido's code they use another way to
    # normalize the data which we decided not to implement.
    subject_image = sklearn.preprocessing.scale(subject_image)
    concat_matrix = concatenate_matrices(current_component,
                                         subject_image)
    cov_mat = concat_matrix @ concat_matrix.transpose()
    # get eigenvalues in ascending order.
    _, v = np.linalg.eigh(cov_mat)

    # get the biggest eigenvectors.
    v = np.fliplr(v)[:, : min(v.shape[0], iterative_n)]

    return v.transpose() @ concat_matrix


def get_subject_dir(subjects_rfMRI_folder,
                    subject):
    return os.path.join(subjects_rfMRI_folder, subject, 'MNINonLinear/Results/')


def get_subject_rest_rfMRI_image(subject_dir, session_n, session_type):
    file_name = os.path.join(
        subject_dir,
        # TODO(loya) move to a pretty format without these f's
        'rfMRI_REST{session_n}_{session_type}/' +
        'rfMRI_REST{session_n}_{session_type}' +
        '_Atlas_hp2000_clean.dtseries.nii')
    cifti, _ = utils.cifti_utils.load_nii_brain_data_from_file(file_name)
    return np.asarray(cifti, dtype=dtype)


def iterative_pca(subjects, iterative_n=1200, result_n=1000):
    res = None
    for subject in subjects:
        for session in subject.sessions:
            res = iterative_pca_subject(
                res,
                session.image,
                iterative_n)
    return res[:result_n, :].transpose()


def iterative_pca_from_files(
        subjects_rfMRI_folder,
        subjects,
        iterative_n=1200,
        result_n=1000):
    # Loop over sessions and subjects
    pca_res = None

    for a in ['1', '2']:
        for b in ['LR', 'RL']:
            for subject in subjects:
                subject_dir = \
                    get_subject_dir(subjects_rfMRI_folder, subject)
                subject_session_image = \
                    get_subject_rest_rfMRI_image(subject_dir, a, b)
                pca_res = iterative_pca_subject(
                    pca_res,
                    subject_session_image,
                    iterative_n)
    return pca_res[:result_n, :].transpose()

    # # % Save group PCA results
    # # dt = utils.cifti_utils.load_nii_brain_image_from_file('./extras/CIFTIMatlabReaderWriter/example.dtseries.nii')
    # # dt = data
    # print(f'Saving PCA results  {data.shape[0]} x {data.shape[1]}')
    # result_path = os.path.join(
    #     outdir, f'GROUP_PCA_{outname}_RFMRI.dtseries.nii')
    # utils.cifti_utils.save_image_to_file(data, result_path)
