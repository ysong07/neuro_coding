import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import delay_map_clustering as dmc
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


dst = scipy.spatial.distance.euclidean


def gap(data, refs=None, nrefs=20, ks=range(1,11)):
    """
        Compute the Gap statistic for an nxm dataset in data.
        
        Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
        or state the number k of reference distributions in nrefs for automatic generation with a
        uniformed distribution within the bounding box of data.
        
        Give the list of k-values for which you want to compute the statistic in ks.
        """
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        
            
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
                    
    gaps = scipy.zeros((len(ks),))
    std = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        k_means = cluster.KMeans(n_clusters=k)
        k_means.fit(data)
        kml = k_means.labels_
        kmc = k_means.cluster_centers_
#        (kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])
                        
        refdisps = scipy.zeros((rands.shape[2],))
        log_refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            k_means = cluster.KMeans(n_clusters=k)  
            k_means.fit(data)
            kml = k_means.labels_
            kmc = k_means.cluster_centers_
#            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
            log_refdisps[j] = scipy.log(refdisps[j])
        
        std[i] = np.std(log_refdisps)*(np.sqrt(1+1/nrefs))
        gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
    return gaps, std
if __name__ == '__main__':
#    raw_correlation_matrix = dmc.load_raw_correlation()
##    scipy.cluster.vq.kmeans2(raw_correlation_matrix, 2)
#    gaps, std = gap(raw_correlation_matrix,refs=None,nrefs=50,ks=range(2,30))
    """dm_avg_n, dm_max_n, dm_avg_un, dm_max_un = dmc.load_delay_maps(delay_map_str='delay_map_cleaned.mat')
        avg_n, max_n, avg_un, max_un = dmc.load_2d_delay_maps(dm_avg_n,
        dm_max_n,
        dm_avg_un,
        dm_max_un)"""
    delay, energy, mask, after_pca, eigen, starting, ending, indices = dmc.load_features_and_delay_map()
    fit_x, fit_y, orig_x, orig_y = dmc.load_centroids_of_data()
    fit_xy = np.append(fit_x, fit_y, axis=1)
    
    raw_correlation_matrix = dmc.load_raw_correlation()
    
    
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
    print np.shape(feature_matrix)
    gaps, std = gap(feature_matrix,refs=None,nrefs=500,ks=range(2,35))
    
    print gaps
    print std
