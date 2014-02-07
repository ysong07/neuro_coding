from sklearn import metrics
import scipy.io
import os
import numpy as np
from sklearn import mixture
from sklearn.hmm import GaussianHMM
from sklearn import cluster
import math
import delay_map_clustering as dmc
import scipy.cluster.vq as scivq

def dirichlet_process_clustering(label = 'labeled_cluster',feature='all_correlation',feature2= 1,n_component=1,ct='diag',n_iter =100):
    a = scipy.io.loadmat(label)
    labeled_C = a['labeled_cluster']
    b = scipy.io.loadmat(feature)
    feature_map = b['intra_corr']
    c = scipy.io.loadmat('all_correlation_trajectory_ridge')
    feature_map3 = c['intra_corr']
    score_out_aic = []
    score_out_bic = []
    for jj in range(14):
        array_Compare = labeled_C[jj,:]
        indices = dict()
        score = dict()
        if (jj<=6):
            feature_map2 = feature2
            for ii in set(array_Compare):
                indices[ii] = np.where(array_Compare == ii)
                if (len(indices[ii][0])>1):
                    clf = mixture.GMM(n_init=10)
                    X = feature_map2[indices[ii][0]]
                    clf.fit(X)
                    vector_size = float(indices[ii][0].size)
                    abc = math.log(vector_size/508)
                    score[ii] = sum(clf.score(X))+abc*indices[ii][0].size
                else:
                    score[ii] = math.log(float(1)/508)
            # diag covariance matrix, mean, prior distribution = 508*2+ score.size
            #    print score
            score_out_aic.append(2*(720*2+ len(score))-2*sum(score.itervalues()))
            # aic 2*k - 2*ln(P) bic : k*ln(n)-2ln(P) n:508 for correlation matrix
            # n = 720?200? for delay
            score_out_bic.append((720*2+ len(score))*math.log(508)-2*sum(score.itervalues()))
        elif (jj<12):
            for ii in set(array_Compare):
                indices[ii] = np.where(array_Compare == ii)
                if (len(indices[ii][0])>1):
                    clf = mixture.GMM(n_init=10)
                    X = feature_map[indices[ii][0]]*10
                    clf.fit(X)
                    vector_size = float(indices[ii][0].size)
                    abc = math.log(vector_size/508)
                    score[ii] = sum(clf.score(X))+abc*indices[ii][0].size
                else:
                    score[ii] = math.log(float(1)/508)
# diag covariance matrix, mean, prior distribution = 508*2+ score.size
#    print score
            score_out_aic.append(2*(508*2+ len(score))-2*sum(score.itervalues()))
# aic 2*k - 2*ln(P) bic : k*ln(n)-2ln(P) n:508 for correlation matrix
    # n = 720?200? for delay
            score_out_bic.append((508*2+ len(score))*math.log(508)-2*sum(score.itervalues()))
        else:
            for ii in set(array_Compare):
                indices[ii] = np.where(array_Compare == ii)
                if (len(indices[ii][0])>1):
                    clf = mixture.GMM(n_init=10)
                    X = feature_map3[indices[ii][0]]*10
                    clf.fit(X)
                    vector_size = float(indices[ii][0].size)
                    abc = math.log(vector_size/508)
                    score[ii] = sum(clf.score(X))+abc*indices[ii][0].size
                else:
                    score[ii] = math.log(float(1)/508)
            # diag covariance matrix, mean, prior distribution = 508*2+ score.size
            #    print score
            score_out_aic.append(2*(508*2+ len(score))-2*sum(score.itervalues()))
            # aic 2*k - 2*ln(P) bic : k*ln(n)-2ln(P) n:508 for correlation matrix
            # n = 720?200? for delay
            score_out_bic.append((508*2+ len(score))*math.log(508)-2*sum(score.itervalues()))

            

    return indices,score_out_aic,score_out_bic


if __name__ == '__main__':
    

    
    # a = scipy.io.loadmat('labeled_cluster')
    #labeled_C = a['labeled_cluster']
    #array_Compare = labeled_C[6,:]
    #metric = [0 for x in range(14)]
    #for i in range(14):
    #   array_method = labeled_C[i,:]
    #   metric[i] = metrics.v_measure_score(array_Compare, array_method)
    #print metric
    

    delay, energy, mask, after_pca, eigen, starting, ending, indices = dmc.load_features_and_delay_map()
    fit_x, fit_y, orig_x, orig_y = dmc.load_centroids_of_data()
    fit_xy = np.append(fit_x, fit_y, axis=1)
    
    raw_correlation_matrix = dmc.load_raw_correlation('all_correlation_trajectory_ridge.mat')
    all_correlation = dmc.load_raw_correlation('all_correlation_trajectory_ridge.mat')
    
    
    """Resampled trajectories"""
    ####### Cluster between 10 and 20 frames
    cluster_10_20_x, cluster_10_20_y, cluster_10_20_label = dmc.load_resampled_trajectories()
    cluster_10_20_xy = np.concatenate((cluster_10_20_x,cluster_10_20_y),axis=1)
    
    # Matlab indices to Python Indices
    indices -= 1
    n_pca_comp = 200
    
    r_delay = delay[:,:,indices.T][:,:,:,0]
    r_energy = energy[:,:,indices.T][:,:,:,0]
    r_mask = mask[:,:,indices.T][:,:,:,0]
    
    r_delay_2d = np.reshape(r_delay, (r_delay.shape[0]*r_delay.shape[1],
                                      r_delay.shape[2]))
    r_energy_2d = np.reshape(r_energy, (r_energy.shape[0]*r_energy.shape[1],
                                        r_energy.shape[2]))
    r_mask_2d = np.reshape(r_mask, (r_mask.shape[0]*r_mask.shape[1],
                                    r_mask.shape[2]))
    """
        r_total_2d = np.append(r_delay_2d, r_energy_2d, axis=0)
        r_total_2d = np.append(r_total_2d, r_mask_2d, axis=0)
        """
    feature_matrix = after_pca[:n_pca_comp,indices.T][:,:,0].T
    
    ### New Delay, Energy and Binary Map
    delay_shift = True
    delay_n, energy_n, binary_n = dmc.load_delay_energy_binary_map()
    thre_max = 70
    thre_min = -30
    delay_n[delay_n > thre_max] = thre_max
    delay_n[delay_n < thre_min] = thre_min
    
    #### FOR PCA Construction
    
    delay_p = np.reshape(delay_n, (delay_n.shape[0]*delay_n.shape[1], delay_n.shape[2])).T
    energy_p = np.reshape(energy_n, (energy_n.shape[0]*energy_n.shape[1], energy_n.shape[2])).T
    binary_p = np.reshape(binary_n, (binary_n.shape[0]*binary_n.shape[1], binary_n.shape[2])).T
    total_p = np.concatenate((delay_p, energy_p, binary_p), axis=1)
#    total_pca_comb_features, eigen_pca_comb = get_pca_components(total_p)
#    pca_features_temp = total_pca_comb_features[indices,:]
#    pca_features = np.reshape(pca_features_temp, (pca_features_temp.shape[1], pca_features_temp.shape[2]))
    
    
#    pca_raw_features, eigen_pca_comb  = get_pca_components(raw_correlation_matrix);
    
    
    
    #indices = dmc.load_indices()
    delay = delay_n[:,:,indices]
    delay = np.reshape(delay, (delay.shape[0], delay.shape[1], delay.shape[3]))
    delay_t = delay.T
    n_delay = np.reshape(delay_t, (delay_t.shape[0], delay_t.shape[1]*delay_t.shape[2]))
    if delay_shift:
        for d in n_delay:
            d -= np.min(d)
    
    energy = energy_n[:,:,indices]
    energy = np.reshape(energy, (energy.shape[0], energy.shape[1], energy.shape[3]))
    energy_t = energy.T
    n_energy = np.reshape(energy_t, (energy_t.shape[0], energy_t.shape[1]*energy_t.shape[2]))
    binary = binary_n[:,:,indices]
    binary = np.reshape(binary, (binary.shape[0], binary.shape[1], binary.shape[3]))
    binary_t = binary.T
    n_binary = np.reshape(binary_t, (binary_t.shape[0], binary_t.shape[1]*binary_t.shape[2]))
    total_features = np.concatenate((n_delay, n_energy, n_binary), axis=1)
    
    ### WHITENED DATA
    constantInfinity = 10000
    w_delay = scivq.whiten(n_delay.T)
    w_delay = w_delay.T
    w_energy = scivq.whiten(n_energy.T)
    w_energy = w_energy.T
    w_binary = scivq.whiten(n_binary.T)
    w_binary = w_binary.T
    w_binary[w_binary == np.infty] = constantInfinity
    w_delay_energy = np.concatenate((w_delay, w_energy), axis=1)
    w_energy_binary = np.concatenate((w_energy, w_binary), axis=1)
    w_delay_binary = np.concatenate((w_delay, w_binary), axis=1)
    w_feature_combination = np.concatenate((w_delay, w_energy, w_binary), axis=1)
    
    ### PCA DATA
#    pca_delay,_ = get_pca_components(n_delay)
#    pca_energy,_ = get_pca_components(n_energy)
#    pca_binary,_ = get_pca_components(n_binary)
#    pca_sep_combination = np.concatenate((pca_delay, pca_energy, pca_binary), axis=1)
#    
    """"
        whiten = False
        delay_pca, n_comp_delay = get_pca_components(r_delay_2d, whiten=whiten)
        energy_pca, n_comp_energy = get_pca_components(r_energy_2d, whiten=whiten)
        mask_pca, n_comp_mask = get_pca_components(r_mask_2d, whiten=whiten)
        total_pca, n_comp_total = get_pca_components(r_total_2d, whiten=whiten)
        
        delay_features = delay_pca.T[:,:n_comp_delay]
        energy_features = energy_pca.T[:,:n_comp_energy]
        mask_features = mask_pca.T[:,:n_comp_mask]
        total_features = total_pca.T[:, :n_comp_total]
        
        fit_xy_t = fit_xy.T
        trajectory_pca, n_comp_traj = get_pca_components(fit_xy_t)
        trajectory_pca, n_comp_traj = get_pca_components(fit_xy_t)
        trajectory_features = trajectory_pca[:n_comp_traj,:]
        trajectory_features = trajectory_features.T
        
        trajectory_features = fit_xy
        
        temp = np.append(delay_features, energy_features, axis=1)
        combination_features = np.append(temp, mask_features, axis=1)
        """
    clustering_algos = ['k_delay', 'ms_delay', 'dp_delay', 'hmm_delay',
                        'k_centroid', 'ms_centroid', 'dp_centroid', 'hm_centroid',
                        'model_selection','evaluation', 'intra_inter_plot','k_value_determination',
                        'clustering_index_coloring_for_clusters' ]
    data_feature = np.concatenate((w_delay, w_energy), axis=1)
    print data_feature.shape

indices,score_out_aic,score_out_bic = dirichlet_process_clustering(label = 'labeled_cluster',feature='all_correlation',feature2 = data_feature ,n_component=1,ct='diag',n_iter =2)
print indices
print 'aic'
print score_out_aic
print 'bic'
print score_out_bic





