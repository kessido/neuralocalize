"""This code simulates the feature extraction part of the connectivity model.
"""

import sklearn
import scipy
import numpy as np
import src.utils.utils as util
import mdp
import nibabel

ICA_FUCKING_CONST = 0.00000117915

# TODO(loya) make sure and remove these two
import numpy.matlib as matlib
import sklearn.decomposition


def run_group_ica_separately(cifti_image,BM,threshold=ICA_FUCKING_CONST, num_ic=40, N=91282):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for each hemisphere separately
    :param left_hemisphere_data:
    :param right_hemisphere_data:
    :param BM:
    :param num_ic: number of independent components
    :param threshold:
    :return:
    """
    #TODO (Itay) cifti_image to left_hemisphere_data and right_hemisphere_data
    left_hemisphere_data = cifti_extract_data(cifti_image,BM,'L')
    right_hemisphere_data = cifti_extract_data(cifti_image,BM,'R')
    #np.transpose(left_hemisphere_data)
    #np.transpose(right_hemisphere_data)
    # Compute ICA
    left_ica,_,_ = sklearn.decomposition.fastica(left_hemisphere_data,num_ic)
    right_ica,_,_ = sklearn.decomposition.fastica(right_hemisphere_data,num_ic)
    #flip signs (large tail on the right)
    #left_ica = np.multiply(left_ica,
    #                    np.tile(np.sign(np.sum(np.sign(np.multiply(left_ica, (np.abs(left_ica) > ICA_FUCKING_CONST).astype(float))), 1))
    #                        ,(1, left_ica.shape[1])))
    #right_ica = np.multiply(right_ica,
    #                    np.tile(np.sign(np.sum(np.sign(np.multiply(right_ica, (np.abs(right_ica) > ICA_FUCKING_CONST).astype(float))), 1))
    #                        ,(1, right_ica.shape[1])))
    thresh = (np.abs(left_ica) > ICA_FUCKING_CONST).astype(float)
    ica_time_thresh = np.multiply(left_ica, thresh)
    end_res = np.sign(np.sum(np.sign(ica_time_thresh), 1))
    end_res_t = np.reshape(end_res, (num_ic, 1))
    tile_res = np.tile(end_res_t, (1, left_ica.shape[1]))
    left_ica = np.multiply(left_ica, tile_res)

    thresh = (np.abs(right_ica) > ICA_FUCKING_CONST).astype(float)
    ica_time_thresh = np.multiply(right_ica, thresh)
    end_res = np.sign(np.sum(np.sign(ica_time_thresh), 1))
    end_res_t = np.reshape(end_res, (num_ic, 1))
    tile_res = np.tile(end_res_t, (1, right_ica.shape[1]))
    right_ica = np.multiply(right_ica, tile_res)


    #keep ICA components that have L/R symmetry
    #left-right DICE of cortical ICs to
    #1) re-order the ICs
    #2) select the ICs that are found in both hemispheres
    x = np.zeros([32492, left_ica.shape[0]])
    y = np.zeros([32492, right_ica.shape[0]])
    np.put(x, BM[0].surface_indices, np.transpose(left_ica))
    np.put(y, BM[1].surface_indices, np.transpose(right_ica))
    D = dice(x>threshold,y>threshold)
    D_threshold = (D == np.tile(D.max(1), (D.shape[1],1))).astype(np.float32)
    D_tmp = ((D*D_threshold) == np.tile(np.amax(D*D_threshold,axis=0),(D.shape[1],1)))
    D_threshold = D_tmp*D_threshold

    #TODO(Itay) just for tests, delete the next line:
    D_threshold[np.ix_(list(range(0,36)),[0])] = np.ones((36,1))
    #just for test, delete the prev line

    r = np.nonzero(np.sum(D_threshold,1))[0]
    c = D_threshold.argmax(1)
    c = c[r]
    #save
    x = np.zeros((N,len(r)))
    x[np.ix_(list(BM[0].data_indices),list(range(x.shape[1])))] = np.transpose(left_ica[r,:])
    x[np.ix_(list(BM[1].data_indices), list(range(len(c))))] = np.transpose(right_ica[c, :])
    #x[np.ix_(list(BM[1].data_indices), [0, len(r) - 1])] = np.transpose(right_ica[c, :])
    return np.transpose(x)

def cifti_extract_data(cifti_image,BM,side):
    '''extracts data from cifti images'''
    if (side == 'L'):
        indices = BM[0].data_indices
        data = cifti_image[:,indices.start:indices.stop]
    else :
        if (side == 'R'):
            indices = BM[1].data_indices
            data = cifti_image[:, indices.start:indices.stop]
        else:
            if (side == 'both'):
                data = cifti_image
            else:
                print('error: bad cifti_extract_data command, side is not L, R or both')
    return data

def dice(x,y):
    """gets x = N*nx and y = N*ny and return nx*ny
    :param x should be N*nx:
    :param y should be N*ny:
    :return: nx*ny
    """
    if (x.shape[0] != y.shape[0]):
        print('x and y incompatible (dice)')
    nx = x.shape[1]
    ny = y.shape[1]
    xx = np.tile(np.sum(x,0),(ny,1))
    yy = np.tile(np.sum(y,0), (nx,1))
    temp = np.dot(np.transpose(x.astype(np.float64)),y.astype(np.float64))
    res = 2 * np.divide(temp, xx+yy)
    return res


def run_group_ica_together(cifti_image,BM, num_ic=50):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for both hemispheres, to use as spatial filters.
    :param both_hemisphere_data:
    :param num_ic:
    :return:
    """
    both_hemisphere_data = cifti_extract_data(cifti_image,BM,'both')
    #np.transpose(both_hemisphere_data)
    both_ica,_,_ = sklearn.decomposition.fastica(both_hemisphere_data,num_ic)
    thresh = (np.abs(both_ica) > ICA_FUCKING_CONST).astype(float)
    ica_time_thresh = np.multiply(both_ica, thresh)
    end_res = np.sign(np.sum(np.sign(ica_time_thresh), 1))
    end_res_t = np.reshape(end_res, (num_ic, 1))
    tile_res = np.tile(end_res_t, (1, both_ica.shape[1]))
    both_ica = np.multiply(both_ica, tile_res)

    return np.transpose(both_ica)



def run_dual_regression(left_right_hemisphere_data, BM, subjects, size_of_g=91282):
    """Runs dual regression TODO(whoever) expand and elaborate.

    :param left_right_hemisphere_data:
    :param subjects:
    :param size_of_g:
    :return:
    """

    single_hemisphere_shape = left_right_hemisphere_data.shape[1]
    print(single_hemisphere_shape)
    G = np.zeros([size_of_g, single_hemisphere_shape * 2])
    hemis = np.zeros([size_of_g, single_hemisphere_shape * 2])

    G[BM[0].data_indices, :single_hemisphere_shape] = left_right_hemisphere_data[BM[0].data_indices, :]
    G[BM[1].data_indices, single_hemisphere_shape: 2 * single_hemisphere_shape] = left_right_hemisphere_data[
                                                                                           BM[1].data_indices, :]

    hemis[BM[0].data_indices, :single_hemisphere_shape] = 1
    hemis[BM[1].data_indices, single_hemisphere_shape : 2 * single_hemisphere_shape] = 1

    g_pseudo_inverse = np.linalg.pinv(G)
    for subject in subjects:
        subject_data = []
        print("NUM OF SESSIONS:", len(subject.sessions))
        for session in subject.sessions:
            normalized_cifti = sklearn.preprocessing.scale(session.cifti, with_mean=False)
            deterended_data = np.transpose(scipy.signal.detrend(np.transpose(normalized_cifti)))
            subject_data.append(deterended_data)
        # TODO(loya) this is a potential bug. Stack? squeeze? concatenate? tile?
        subject_data = np.concatenate(subject_data, axis=1)
        T = g_pseudo_inverse @ subject_data

        t = util.fsl_glm(np.transpose(T), np.transpose(subject_data))
        cifti_data = np.transpose(t) * hemis
        return cifti_data

def get_subcortical_parcellation(cifti_image, brain_maps):
    """Get sub-cortical parcellation using atlas definitions and current data.
    :return: (no. voxel, cortical parcellation parts)
    """

    def do_nothing_brain_map_handler(*args):
        pass

    def use_as_is_brain_map_handler(cifti_image, current_map):
        """Uses the brain map with no prepossessing
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels, 1), 1 if index is in part of the current part, 0 otherwise
        """
        ret = np.zeros([cifti_image.shape[1], 1])
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        ret[start_index:end_index] = 1
        return ret

    def corrcoef_and_spectral_ordering(mat):
        """Implementation of reord2 + corrcoef function that was used in the matlab version
        :param mat: The data matrix
        :return: spectral ordering of the corrcoef matrix of A
        """
        mat = np.corrcoef(mat.transpose()) + 1
        ti = np.diag(np.sqrt(1. / np.sum(mat, 0)))
        W = np.matmul(np.matmul(ti, mat), ti)
        U, S, V = np.linalg.svd(W)
        S = np.diag(S)
        P = np.multiply(np.matmul(ti, np.reshape(U[:, 1], [U.shape[0], 1])), np.tile(S[1, 1], (U.shape[0], 1)))
        return P

    def half_split_using_corrcoef_and_spectral_ordering_brain_map_handler(cifti_image, current_map):
        """This split the data into 2 different clusters using corrcoef,
        spatial ordering, and positive\negative split
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels , 2), each vector is a 0\1 vector representing the 2 clusters
        """
        res = np.zeros([cifti_image.shape[1], 2])
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        cifti_current_map_data = cifti_image[:, start_index:end_index]
        spatial_ordering = corrcoef_and_spectral_ordering(cifti_current_map_data)
        res[start_index:end_index, :] = np.hstack((spatial_ordering > 0, spatial_ordering < 0)).astype(float)
        return res

    def label_to_function(label):
        label = label.rsplit('_', 1)[0]
        labels_to_function = {
            'CIFTI_STRUCTURE_CORTEX': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_ACCUMBENS': use_as_is_brain_map_handler,
            'CIFTI_STRUCTURE_AMYGDALA': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_BRAIN': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_CAUDATE': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_CEREBELLUM': ica_clustering_brain_map_handler,
            'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_HIPPOCAMPUS': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_PALLIDUM': use_as_is_brain_map_handler,
            'CIFTI_STRUCTURE_PUTAMEN': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_THALAMUS': ica_clustering_brain_map_handler,
        }
        return labels_to_function[label]

    def ica_clustering_brain_map_handler(cifti_image, current_map):
        """This split the data into 3 parts by counting the eddect of each of the 3 first components
        in the ICA analysis on all the voxels, and determines the cluster by the one with the maximum
        connection to the voxel.
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels , 3), each vector is a 0\1 vector representing the 3 clusters
        """
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        cifti_current_map_data = cifti_image[:, start_index:end_index]
        # todo(kess) this FastICA does not yield the same result as
        ica_Y, _, _ = sklearn.decomposition.fastica(cifti_current_map_data, 3,)
        # ica_Y = np.multiply(ica_Y,
        #                     np.tile(np.reshape(
        #                         np.sign(np.sum(np.sign(np.multiply(ica_Y, (np.abs(ica_Y) > ICA_FUCKING_CONST).astype(float))), 1)),
        #                         (3, 1)), (1, ica_Y.shape[1])))

        thresh = (np.abs(ica_Y) > ICA_FUCKING_CONST).astype(float)
        ica_time_thresh = np.multiply(ica_Y, thresh)
        end_res = np.sign(np.sum(np.sign(ica_time_thresh),1))
        end_res_t = np.reshape(end_res,(3,1))
        tile_res = np.tile(end_res_t, (1, ica_Y.shape[1]))
        ica_Y = np.multiply(ica_Y, tile_res)

        res = np.zeros([cifti_image.shape[1], ica_Y.shape[0]])
        res[start_index:end_index, :] = ica_Y.transpose()
        return res

    sub_cortex_clusters = []
    for current_map in brain_maps:
        x = label_to_function(current_map.brain_structure)(cifti_image, current_map)
        if x is not None:
            sub_cortex_clusters.append(x)
    return np.hstack(sub_cortex_clusters).transpose()


def get_semi_dense_connectome(semi_dense_connectome_data, subjects):
    """Final feature extraction (forming semi-dense connectome)
    For each subject, load RFMRI data, then load ROIs from above to calculate semi-dense connectome.

    # ASSUMES:
    # getting a subject list holding session array sessions, each holding left h. data, right h. data, ROIs, BM and CIFTI.
    # In MATLAB they're all being loaded. All these members are assumed to be numpy arrays.
    # (This is handled as an object but can be changed to a list of tuples or dictionary, whatever)

    :return: A dictionary from a subject to its correlation coeff.
    """
    subject_to_correlation_coefficient = {}
    # TODO(loya) shapes must be validated.
    for subject in subjects:
        W = []
        print ("SHAPES:",
               "left_right_hemisphere_data:", subject.left_right_hemisphere_data.shape,
               "semi_dense_connectome_data:", semi_dense_connectome_data.shape)
        ROIS = np.concatenate([subject.left_right_hemisphere_data, semi_dense_connectome_data], axis=1)
        print("ROIS.shape:", ROIS.shape)
        for session in subject.sessions:
            W.append(sklearn.preprocessing.scale(session.cifti))
        # TODO(loya) this might cause a bug.
        print("W[0].shape before concat:", W[0].shape)
        W = np.concatenate(W, axis=1)
        print("W SHAPE AFTER CONCAT:", W.shape)
        # MULTIPLE REGRESSION
        # TODO(loya) there was a transpose here that caused a bug. removed because for now, might be a problem.
        print("ROIS pinv shape:", np.linalg.pinv(ROIS).shape)
        T = np.linalg.pinv(ROIS) @ W
        print("T shape:", T.shape)
        # CORRELATION COEFFICIENT
        F = sklearn.preprocessing.normalize(T, axis=1) @ np.transpose(sklearn.preprocessing.normalize(W, axis=0))
        print("F.shape:", F.shape)
        subject_to_correlation_coefficient[subject] = F
    return subject_to_correlation_coefficient


def extract_features(cifti, brain_maps, pca):
    # TODO return end point subj_RFMRI_nosmoothing.dtseries.nii
    pass


def get_spatial_filters(pca):
    # %% Load spatial filters
    # % then threshold and do a winner-take-all
    # disp('Load Filters');
    # filters = open_wbfile([outdir '/ica_both_lowdim.dtseries.nii']);
    # [m,wta]=max(filters.cdata,[],2);
    # wta = wta .* (m>2.1);
    # S = zeros(size(filters.cdata));
    # for i=1:size(filters.cdata,2)
    #     S(:,i) = double(wta==i);
    # end
    pass
