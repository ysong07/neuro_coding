from time import time

import numpy as np
import pylab as pl
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection,mixture)
from sklearn.hmm import GaussianHMM
import delay_map_clustering as dmc
import math
import clustering_delay_maps as cdm
import scipy.io as scipy_io
import scipy.cluster.vq as scivq

def 
if __name__ == '__main__':
    #DP
    # dp_components = 25 and alpha = 10 => best so far in terms of coherence
    # alpha => 1 and .1 is also quite good
    dp_components = (30, 20)
    alpha = (0.1, 1)
    n_iter = 10000
    n_neighbors = 30
    delay,energy,mask = dmc.load_data_40_maps()
    all_correlation = dmc.load_raw_correlation('all_correlation_40.mat')
    X = all_correlation
    X_iso = manifold.Isomap(n_neighbors, n_components=5).fit_transform(X)
#    w_correlation = scivq.whiten(X_iso);
    
#    print(w_correlation.shape)
    (aic, bic), dp_indices,labels = cdm.dirichlet_process_clustering(X_iso, dp_components[0], a=alpha[0], n_iter=n_iter)
    scipy_io.savemat('data_40_Dirichlet_correlation_isomap_label',{'labels':labels})
    print(labels.shape)
