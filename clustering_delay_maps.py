import numpy as np
from sklearn import cluster
import delay_map_clustering as dmc
import matplotlib.pyplot as plt
import matplotlib as mlp
import itertools
from scipy import linalg
import scipy.io as scipy_io
import scipy.cluster.vq as scivq
from sklearn import mixture
from sklearn.hmm import GaussianHMM
import datetime
import os
import matplotlib.mlab as mlab
import cPickle
import math
figure_path = 'ms_thesis_figures'
now = datetime.datetime.now()
today = now.strftime("%Y-%m-%d")
fig_path = os.path.join(figure_path, today)

def intra_over_inter(indices,all_correlation):
    cluster = dict()
    num_cluster = len(set(indices))
    for ii in set(indices):
        cluster[ii] = np.where(indices ==ii)
        cluster_corr = np.zeros([ii+1,ii+1])
    inter_norm = np.zeros([ii+1,ii+1])
    intra_norm = 0
    aver_corr = np.empty(ii+1)

    for i in cluster.keys():
        for j in cluster.keys():
            if i != j:
                inter_matrix1 = all_correlation[cluster[i]]
                inter_matrix2 = inter_matrix1[:,cluster[j]]
                cluster_corr[i,j] = inter_matrix2.mean()
                inter_norm[i,j] = np.size(cluster[i])*np.size(cluster[j])
            else:
                inter_norm[i,j] = 0
                inter_matrix1 = all_correlation[cluster[i]]
                inter_matrix2 = inter_matrix1[:,cluster[j]]
                cluster_corr[i,j] = inter_matrix2.mean()
                intra_norm = intra_norm + np.size(cluster[i])*np.size(cluster[i]) -np.size(cluster[i])
                if np.size(cluster[i])!=1:                    
                    aver_corr[i] = (inter_matrix2.sum() - np.size(cluster[i]))
                else:
                    aver_corr[i] = 1
    aver_intra = aver_corr.sum()/ intra_norm
    aver_inter = np.multiply(cluster_corr,inter_norm).sum()/(inter_norm.sum())
    return aver_intra/aver_inter,num_cluster


def get_pca_components(data_2d, x_component=1, whiten=False):
        """data_2d => shape(n_samples, n_features)
        """
        pca_output = mlab.PCA(data_2d)
        n_component = get_eigenvalue_number_by_percentage(pca_output.fracs) + x_component
        return pca_output.Y[:,:n_component], pca_output.fracs

def get_eigenvalue_number_by_percentage(eigen_values, percent=.995):
    eigen_sum = np.sum(eigen_values)
    enough = eigen_sum * percent
    temp = 0
    for ii, eigen in enumerate(eigen_values):
        temp += eigen
        if temp >= enough:
            return ii
    return eigen_values.size


def generate_indices(labels, n_clusters=10):
    indices = dict()
    for ii in set(labels):
        indices[ii] = np.where(labels == ii)
    return indices

def k_means_clustering(data, n_clusters=10):
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(data)
    labels = k_means.labels_
    indices = generate_indices(labels, n_clusters=n_clusters)
    return indices

def get_intravariance_cluster(main_data, indices):
    """ Returns the intravariance of the clusters indexed in indices,
    it also returns the average of the component inside of one cluster (avg_dict)
    it returns all of the variance in a dictionary given their cluster for Davies-Bouldin
    Index computation"""
    mse_dict = dict()
    avg_dict = dict()
    for cluster, members in indices.iteritems():
        temp = main_data[members,:]
        temp = np.reshape(temp, (temp.shape[1], temp.shape[2]))
        n_members = np.size(members)
        average = np.sum(temp,  axis=0) / n_members
        avg_dict[cluster] = average
        error = temp - average
        error = error.astype(np.int64, copy=False)
        mse = np.sum(error * error)
        mse_dict[cluster]  = mse
        print("Mean Square error for cluster {}: {} ".format(cluster, mse))
    return (sum(mse_dict.values()) / float(len(mse_dict))), mse_dict, avg_dict

def get_intervariance_cluster(avg_dict, max_i=False):
    """ Gets and average vector dictionary and returns the intervariance
    cluster of the clustering space either getting the maximum of the intervariance
    or the average of the intervariance for a given cluster.
    """
    inter_variance_cluster = dict()
    inter_result = dict()
    for main_cluster, main_average in avg_dict.iteritems():
        for second_cluster, second_average in avg_dict.iteritems():
            if not main_cluster == second_cluster:
                if main_cluster not in inter_variance_cluster:
                    inter_variance_cluster[main_cluster] = {}
                if second_cluster not in inter_variance_cluster[main_cluster]:
                    inter_variance_cluster[main_cluster][second_cluster] = np.sum(np.square(main_average - second_average))
    for k,v in inter_variance_cluster.iteritems():
        if not max_i:
            inter_result[k] = np.sum(v.values()) / len(v)
        else:
            inter_result[k] = np.max(v.values())
    return inter_result, inter_variance_cluster

def get_dbi(intravariance_dict, intervariance_dict, max_d=True):
    """ Computes the Davies-Bouldin Index
    rij, d_i, db: Davies-Bouldin Index from Wikipedia Page
    db is called Davies Bouldin Index
    """
    r_ij = dict()
    d_i = dict()
    for main_cluster, main_average in intravariance_dict.iteritems():
        for second_cluster, second_average in intravariance_dict.iteritems():
            if not main_cluster == second_cluster:
                if main_cluster not in r_ij:
                    r_ij[main_cluster] = {}
                if second_cluster not in r_ij[main_cluster]:
                    r_ij[main_cluster][second_cluster] = (intravariance_dict[main_cluster] +
                        intravariance_dict[second_cluster])/intervariance_dict[main_cluster][second_cluster]
    for k,v in r_ij.iteritems():
        if max_d:
            d_i[k] = max(v.values())
        else:
            if len(v.values()) == 0:
                d_i[k] = 65536
            else:
                d_i[k] = sum(v.values()) / len(v.values())
    if len(d_i) == 0:
        db = 65536
    else:
        db = sum(d_i.values()) / len(d_i)
    return db




def evaluation_parameters_of_clustering_method(feature_matrix, indices, max_i = False, max_d = True, get_db=True):
    """ Returns intravariance, intervariance, and Davies-Bouldin Index
    for a particular clustering algorithm given the data clustering algorithm run
    and resulting indices of the clustering algorithm"""
    # max_i : Intervariance cluster definition, if it is True, the intervariance resembles to the
    # Davies Bouldin Index
    # max_d : Davies Bouldin Index => True
    intra_variance, intravariance_dict, avg_dict= get_intravariance_cluster(feature_matrix, indices)
    inter_variance, inter_variance_dict = get_intervariance_cluster(avg_dict)
    if get_db:
        db_index = get_dbi(intravariance_dict, inter_variance_dict, max_d=max_d)
    else:
        db_index = None
    return intra_variance, inter_variance, db_index


def visualize_clusters_1d(data_x, data_y, indices, title='None', ext='pdf'):
    for cluster, ind_label in indices.iteritems():
        plt.figure(num=cluster, figsize=(8, 8))
        plt.title('Cluster %d' % cluster)
        label = ind_label[0]
        n_signals = len(label)
        temp = int(np.ceil(np.sqrt(n_signals)))
        for jj in range(n_signals):
            plt.subplot(temp, temp, jj + 1)
            if type(data_x) == dict:
                plt.scatter(data_x[jj], data_y[jj], c=range(data_x[jj].size))
            else:
                plt.scatter(data_x[jj,:], data_y[jj,:], c=range(data_x[0,:].size))
            plt.xticks([])
            plt.yticks([])
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        png_path = os.path.join(fig_path, title) + '_clusterno:_' + str(cluster)
        pdf_path = png_path + '.pdf'
        plt.savefig(pdf_path, bbox_inches=0, format='pdf', dpi=1000)
        #plt.savefig(png_path)
    plt.show()

def visualize_clusters(data, indices, title='None', ext='pdf', max_range=40):
    data = data.astype(float)
    data[data>=max_range] = np.nan
    for cluster, ind_label in indices.iteritems():
        fig = plt.figure(frameon=False, num=cluster, figsize=(8, 8))
        # it passes a tuple, we need to get rid of the tuple part
        # in manipulating the numpy array
        label = ind_label[0]
        n_images = len(label)
        temp = int(np.floor(np.sqrt(n_images-1))) +1
        for jj in range(n_images):
            ax = fig.add_subplot(temp , temp, jj+1)
            cmap = mlp.cm.jet
            cmap.set_bad('w')
            axim = ax.imshow(data[:,:,label[jj]])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            ax.set_title('{label}'.format(label=label[jj]))

        # HORIZONTAL OPTION
        #cbar_ax = fig.add_axes([0.10, 0.05, 0.65, 0.05])
        #fig.colorbar(axim, cax=cbar_ax, orientation='horizontal')
        # VERTICAL OPTION
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(axim, cax=cbar_ax)
        fig.suptitle('Cluster {cluster}: {members} members'.format(cluster=cluster+1, members=n_images))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        png_path = os.path.join(fig_path, title) + '_clusterno:_' + str(cluster)
        pdf_path = png_path + '.pdf'
        fig.savefig(pdf_path, bbox_inches=0, format='pdf', dpi=1000)
        plt.savefig(png_path)
    fig.show()

def mean_shift_clustering(data, bandwidth=None,
    quantile=0.1, random_state=0, bin_seeding=False):
    # 0.5 value is good for bandwidht => produces 50~ clusters with the whole
    # data
    if bandwidth == None:
        bandwidth_estimate = cluster.estimate_bandwidth(data, quantile=quantile,
            random_state=random_state)
    else:
        bandwidth_estimate = bandwidth
    mean_shift = cluster.MeanShift(bandwidth=bandwidth_estimate,
        bin_seeding=bin_seeding)
    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    indices = generate_indices(labels, n_clusters=n_clusters)

    print("number of estimated clusters : %d" % n_clusters)
    return (indices, labels, n_clusters)
    #print("cluster centers are: %d" % cluster_centers)

def dirichlet_process_clustering(X, n_components=30, a=0.1,
                                                    ct='diag', n_iter=100):
    color_iter = itertools.cycle(['r', 'b', 'g', 'c', 'm', 'y', 'k'])
    pred_dict = dict()
    label_dict = dict()

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    fig_path = os.path.join(figure_path, today)

    for ii, (clf, title) in enumerate([
            #(mixture.GMM(n_components=n_components, covariance_type=ct, n_iter=100),
             #"Expectation-maximization"),
            (mixture.DPGMM(n_components=n_components, covariance_type=ct, alpha=a,
                           n_iter=n_iter),
             "Dirichlet Process,alpha=%2.2f" %a)]):
        clf.fit(X)
        # aic and bic evaluation values for the method
        aic = clf.aic(X)
        bic = clf.bic(X)
        print aic , bic
        print ii 
        print(title + "  is completed")
        fig = plt.figure()
        Y_ = clf.predict(X)
        label_dict[title] = Y_
        pred_dict[ii] = Y_
#        for jj, (mean, covar, color) in enumerate(zip(
#                clf.means_, clf._get_covars(), color_iter)):
#            if not np.any(Y_ == jj):
#                continue
#            plt.scatter(X[Y_ == jj, 0], X[Y_ == jj, 1], .8, color=color)
#        plt.title(title)
#        plt.xticks(())
#        plt.yticks(())
#        if not os.path.exists(fig_path):
#            os.makedirs(fig_path)
#        fig.savefig(os.path.join(fig_path, title) +
#            ("_{0}_{1}_{2}_{3}.png".format(title, a, ct, n_iter)))
#    plt.show()
    labels = pred_dict[0]
    n_clusters = len(np.unique(labels))
    print("DP Produces %s number of clusters" % n_clusters)
    indices = generate_indices(labels, n_clusters)
    return (aic, bic), indices,labels

def build_model_selection(X, cv_types=['spherical', 'tied', 'diag', 'full'],
                                           n_components=[10,15, 20, 21, 22, 23, 24, 25],
                                           alphas=[.1, 1, 10, 100]):
    """Returns two dictionaries which have aic and bic
        information for cv_types and n_components
    """
    ab_list = [(i,j,k) for i in alphas for j in cv_types for k in n_components]
    aic_list = list()
    bic_list = list()
    for ii, (alpha, cv_type, n_component) in enumerate(ab_list):
                (aic_temp, bic_temp), _ = dirichlet_process_clustering(X, n_components=n_component, a=alpha,
                                                    ct=cv_type, n_iter=100)
                aic_list.append(aic_temp)
                bic_list.append(bic_temp)
                print("{0} out of {1} for bic and aic are computed".format(ii, len(ab_list)))
    return (aic_list, bic_list)

def visualize_model_selection(lst, alphas, n_components, cv_types, model='BIC'):
    item_per_alpha = len(lst) / len(alphas)
    lst_np = np.array(lst)
    color_iter = itertools.cycle(['c', 'm', 'y','k', 'r', 'g', 'b'])
    bars = []
    for jj, alpha in enumerate(alphas):
        increment = jj * item_per_alpha
        plt.figure()
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components) + .2 * (i - 2)
            bars.append(plt.bar(xpos, abs(lst_np[((i * len(n_components)) + increment):
                                         (((i + 1) * len(n_components)) + increment)]),
                               width=.2, color=color))
            plt.xticks(n_components)
            plt.ylim([lst_np.min() * 1.01 - .01 * lst_np.max(), lst_np.max()])
            plt.title('{0} score per model for alpha: {1}'.format(model, alpha))
            xpos = np.mod(lst_np.argmin(), len(n_components)) + 1 +\
                .2 * np.floor(lst_np.argmin() / len(n_components))
            #plt.text(xpos, lst_np.min() * 0.97 + .03 * lst_np.max(), '*', fontsize=14)
            plt.xlabel('Upper Bound')
            plt.legend([b[0] for b in bars], cv_types, loc=2)
    plt.show()
    return bars

def get_hmm_states(data, n_comp=25, ct = 'diag', n_iter=1000):
    model = GaussianHMM(n_components=n_comp, covariance_type=ct, n_iter=n_iter)
    # Be careful with the angle brackets
    model.fit([data])
    hidden_states = model.predict(data)
    indices = generate_indices(hidden_states, n_clusters=n_comp)
    return indices, hidden_states

def get_hmm_analysis(trajectory_data, n_comp=5, ct='diag', n_iter=1000):
    model = GaussianHMM(n_components=n_comp, covariance_type=ct, n_iter=n_iter)
    model.fit([trajectory_data])
    hidden = model.predict(trajectory_data)
    indices = generate_indices(hidden, n_clusters=n_comp)
    return indices, hidden

def generate_trajectory(orig_x, orig_y):
    if not len(orig_x) == len(orig_y):
        raise Exception
    else:
        for x,y in zip(orig_x.values(),orig_y.values()):
            yield (x,y)

def build_fit_trajectory_matrix(orig_x, orig_y, order=5, n_samples=20, new_positions=None, polar=False):
    """ Generator Object returns a tuple which has x and y coordinates
    of trajectory which may vary in the length"""
    n_trajectory = len(list(generate_trajectory(orig_x, orig_y)))
    x_fit_trajectory = np.zeros((n_trajectory, n_samples))
    y_fit_trajectory = np.zeros((n_trajectory, n_samples))
    generator_object = generate_trajectory(orig_x, orig_y)
    for ii, (x_trajectory, y_trajectory) in enumerate(generator_object):
        coeff_fit = np.polyfit(x_trajectory, y_trajectory, order)
        poly_fit = np.poly1d(coeff_fit)
        if new_positions:
            xt = np.linspace(new_positions[0], new_positions[1], n_samples)
        else:
            xt = np.linspace(np.min(x_trajectory), np.max(x_trajectory), n_samples)
        func_fit = poly_fit(xt)
        if polar:
            for jj,(x,y) in enumerate(zip(xt, func_fit)):
                xt[jj], func_fit[jj] = cartesian_to_polar(x, y)
        x_fit_trajectory[ii,:] = xt
        y_fit_trajectory[ii,:] = func_fit
    return (x_fit_trajectory, y_fit_trajectory)

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def get_average_signal(delay):
    average_signal = list()
    for jj in range(delay.shape[2]):
        average_signal.append(np.average(delay[:,:,jj]))
    return average_signal

def build_cluster_index_signal(indices):
    """ Gets key value pair for cluster, index of the cluster
    returns a new list which has the length of the signal and
    values for the index numbers
    To visualize which cluster occurs most and visualization
    on the 1 dimensional signal"""
    temp = {}
    reverse_d = {}
    index_list = []
    cluster_size = {k:len(v[0].tolist()) for k, v in indices.iteritems()}
    for k,v in indices.iteritems():
        temp[k] = v[0].tolist()
    for cluster, indices in temp.items():
        for index in indices:
            reverse_d[index] = cluster
    for v in reverse_d.values():
        index_list.append(v)
    return index_list, cluster_size

def visualize_average_time_signal_clusters(avg_signal, indices, save=False, title='None'):
    """
    INPUT: avg_signal is 1D signal
                indices has a key,value pair of cluster_number, cluster_indices in the signal
                """
    signal = np.asanyarray(avg_signal)
    num_row_x = num_row_y = int(np.floor(np.sqrt(len(indices)-1))) + 1
    fig, axarr = plt.subplots(num_row_x, num_row_y)
    for k,v in indices.iteritems():
        div = k // num_row_y
        rem = k % num_row_y
        axarr[div, rem].plot(signal[v])
        axarr[div, rem].set_title('Cluster {}'.format(k+1))
        axarr[div, rem].axis('off')
    if save:
        title = os.path.join(figure_path, title) + '.pdf'
        fig.savefig(title)

def visualize_one_sample_from_cluster_delay_map(data, indices, title='None', ext='pdf', max_range=40):
    data = data.astype(float)
    data[data>=max_range] = np.nan
    num_row_x = num_row_y = int(np.floor(np.sqrt(len(indices)-1))) + 1
    fig, axarr = plt.subplots(num_row_x, num_row_y)
    for cluster, ind_label in indices.iteritems():
        members = ind_label[0]
        chosen_delay_map = members[len(members)-1]
        div = cluster // num_row_y
        rem = cluster % num_row_y
        cmap = mlp.cm.jet
        cmap.set_bad('w')
        axim = axarr[div, rem].imshow(data[:,:,chosen_delay_map])
        axarr[div, rem].axis('off')
        axarr[div, rem].set_title('{label}'.format(label=chosen_delay_map))
    png_path = os.path.join(figure_path, title)
    pdf_path = png_path + '.pdf'
    fig.savefig(pdf_path, bbox_inches=0, format='pdf', dpi=1000)


if __name__ == '__main__':
    """ INPUT
    figure_path: where to save the figures
    dm_a** => delay_maps produced by me
    avg_n => 2 dimensional delay maps for clustering, produced by me
    delay, energy, mask => features that will be compared to different clustering algorithms
    """
    figure_path = 'delay_map_figs'
    """dm_avg_n, dm_max_n, dm_avg_un, dm_max_un = dmc.load_delay_maps(delay_map_str='delay_map_cleaned.mat')
    avg_n, max_n, avg_un, max_un = dmc.load_2d_delay_maps(dm_avg_n,
                                                                                               dm_max_n,
                                                                                               dm_avg_un,
                                                                                               dm_max_un)"""
#      data 41
    delay, energy, mask, after_pca, eigen, starting, ending, indices = dmc.load_features_and_delay_map()
 
        
    fit_x, fit_y, orig_x, orig_y = dmc.load_centroids_of_data()
    fit_xy = np.append(fit_x, fit_y, axis=1)
    
    raw_correlation_matrix = dmc.load_raw_correlation()
    all_correlation = dmc.load_raw_correlation()
#

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
    total_pca_comb_features, eigen_pca_comb = get_pca_components(total_p)
    pca_features_temp = total_pca_comb_features[indices,:]
    pca_features = np.reshape(pca_features_temp, (pca_features_temp.shape[1], pca_features_temp.shape[2]))
    
    
    pca_raw_features, eigen_pca_comb  = get_pca_components(raw_correlation_matrix);



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
    pca_delay,_ = get_pca_components(n_delay)
    pca_energy,_ = get_pca_components(n_energy)
    pca_binary,_ = get_pca_components(n_binary)
    pca_sep_combination = np.concatenate((pca_delay, pca_energy, pca_binary), axis=1)

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
    cluster_name = clustering_algos[2]


    ### VISUALIZATION

    thre_max_vis = 40

    delay[delay>thre_max_vis] = thre_max_vis
    #delay -= np.min(delay_p)

    # K-means
    n_clusters = (21
        , 20)
    # For trajectory
    pca_k = False
    # Mean-Shift
    bandwidth = (26, None)
    # For trajectory
    pca_ms = False
    # DP
    # dp_components = 25 and alpha = 10 => best so far in terms of coherence
    # alpha => 1 and .1 is also quite good
    dp_components = (30, 20)
    alpha = (0.015, 1)
    n_iter = 10000
    # For Trajectory
    pca_dpm = False
    order = 5
    n_samples = 20
    new_positions = None
    #new_positions = (1, 10)
    polar = False
    # HMM
    hmm_comp = (13, 5)

    #data_feature = np.concatenate((w_delay, w_energy), axis=1)
    #feature_matrix = data_feature
    #data_feature = np.concatenate((n_delay, n_energy), axis=1)
    data_feature = feature_matrix
    resampled_trajectory = False
    if resampled_trajectory:
        trajectory_x = cluster_10_20_x
        trajectory_y = cluster_10_20_y
        trajectory_xy = cluster_10_20_xy
    else:
        trajectory_x, trajectory_y = build_fit_trajectory_matrix(orig_x, orig_y, order=order,
                                                          n_samples=n_samples, new_positions=new_positions, polar=polar)
        trajectory_xy = np.concatenate((trajectory_x, trajectory_y), axis=1)

    mse_intravariance_cluster = list()
    if cluster_name == clustering_algos[0]:
        k_indices = k_means_clustering(data_feature, n_clusters=n_clusters[0])
        title = 'K-Means_Clustering_with_k={}'.format(n_clusters[0])
        visualize_clusters(delay, k_indices, title)
        """
        k_indices = k_means_clustering(feature_matrix, n_clusters=n_clusters[0])
        title = 'K-Means_Clustering_with_k={}'.format(n_clusters[0])
        visualize_clusters(delay, k_indices, title)"""
        #mse = get_intravariance_cluster(r_delay, k_indices)
        #mse_intravariance_cluster.append(mse)
    elif cluster_name == clustering_algos[1]:
        #ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(pca_features)
        ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(data_feature)
        title = 'Mean-Shift_Clustering_with_ms_n_clusters={}'.format(ms_n_clusters)
        visualize_clusters(delay, ms_indices, title)
        
        
        """
        bandwidth for this featureset is 26
        ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(feature_matrix)
        ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(feature_matrix, bandwidth=bandwidth[0])
        title = 'Mean-Shift_Clustering_with_ms_n_clusters={}'.format(bandwidth[0])
        visualize_clusters(r_delay, ms_indices, title)
        """
    elif cluster_name == clustering_algos[2]:
        
        
        # data 40
        
        delay,energy,mask = dmc.load_data_40_maps()
        input_folder = '/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/'
        input_folder = input_folder + 'all_correlation_40.mat'
        all_correlation = dmc.load_raw_correlation(file)
#        A_a = scipy_io.loadmat('all_correlation_40_clean.mat')
#        A = A_a['A']
#        A_1 = A -1
#        delay = delay[:,A_1][:,:,0]
#        energy = energy[:,A_1][:,:,0]
#
##        print (delay)
#        ### WHITENED DATA
        constantInfinity = 10000
        w_correlation = scivq.whiten(all_correlation.T*100)
        all_correlation = w_correlation.T
        w_delay = scivq.whiten(delay.T)
        w_delay = w_delay.T
        w_energy = scivq.whiten(energy.T)
        w_energy = w_energy.T
        data_feature = np.concatenate((w_delay, w_energy), axis=0)
        print(data_feature.shape)
        
        
        (aic, bic), dp_indices,labels = dirichlet_process_clustering(all_correlation.T, dp_components[0], a=alpha[0], n_iter=n_iter)
        print(labels.shape)
        output_folder = '/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/label_data/'
        output_folder = output_folder+'data_40_Dirichlet_correlation_label'
        scipy_io.savemat(output_folder,{'labels':labels})

#        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(str(alpha[0]).replace(".", ""), dp_components[0])
#        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(str(alpha[0]).replace(".", ""), dp_components[0])
#        scipy_io.savemat('data_40_label',{'labels':labels})

        
#        
#        delay, energy, mask, after_pca, eigen, starting, ending, indices = dmc.load_features_and_delay_map()
#        raw_correlation_matrix = dmc.load_raw_correlation('all_correlation_40.mat')
#        intra_O_inter = np.zeros(50)
#        num_cluster = np.zeros(50)
#        alpha = (0.1, 1)
#        aic = np.zeros(50)
#        bic = np.zeros(50)
#        for i in range(49,50):
#            (aic[i], bic[i]), dp_indices,labels = dirichlet_process_clustering(data_feature, i, a=alpha[0], n_iter=n_iter)
#            all_correlation = dmc.load_raw_correlation('all_correlation.mat')
#            try:
#                intra_O_inter[i],num_cluster[i] = intra_over_inter(labels,all_correlation)
#            except IndexError:
#                intra_O_inter[i] = 0
#                num_cluster[i] = 0
                
#        print type(dp_indices)
#        scipy_io.savemat('Dirchlet_pca_correlation_trajectory_ridge',{'labels':labels,'aver_inter':aver_inter,'aver_intra':aver_intra,'aver_corr':aver_corr})
#        visualize_clusters(delay, dp_indices, title)
        """
        (aic, bic), dp_indices = dirichlet_process_clustering(feature_matrix, dp_components[0], a=alpha[0], n_iter=n_iter)
        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(str(alpha[0]).replace(".", ""), dp_components[0])
        visualize_clusters(delay, dp_indices, title)
        """
    elif cluster_name == clustering_algos[3]:
#        hmm_indices, hmm_states = get_hmm_states(feature_matrix, n_comp=hmm_comp[1])
#        title = 'HMM_Model_component={}'.format(hmm_comp[0])
#        visualize_clusters(r_delay, hmm_indices, title)
        hmm_indices, hmm_states = get_hmm_states(raw_correlation_matrix, n_comp=hmm_comp[0])
        scipy_io.savemat('GaussianHmm_raw_correlation_trajectory',{'labels':hmm_states})
    elif cluster_name == clustering_algos[4]:
        if pca_k:
            trajectory_xy,_ = get_pca_components(trajectory_xy)
        #k_indices = k_means_clustering(trajectory_features, n_clusters=n_clusters[1])
        k_indices = k_means_clustering(trajectory_xy, n_clusters=n_clusters[1])
        title = 'K-Means_Clustering_with_k={}'.format(n_clusters[1])
        """
        if resampled_trajectory:
            x_clu_dict = dict()
            y_clu_dict = dict()
            for ii, row in enumerate(cluster_10_20_x):
                x_clu_dict[ii] = row
            for ii, row in enumerate(cluster_10_20_y):
                y_clu_dict[ii] = row
            resampled_label_dict = dict()
            for key,value in k_indices.iteritems():
                row = value[0]
                resampled_label_dict[key] =cluster_10_20_label[row]
            visualize_clusters_1d(x_clu_dict, y_clu_dict, resampled_label_dict, title)
        else:"""
        visualize_clusters_1d(orig_x, orig_y, k_indices, title)
    elif cluster_name == clustering_algos[5]:
        if pca_ms:
            trajectory_xy,_ = get_pca_components(trajectory_xy)
        ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(trajectory_xy)
        #ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(trajectory_xy, bandwidth=bandwidth[1])
        title = 'Mean-Shift_Clustering_with_ms_n_clusters={}'.format(bandwidth[1])
        visualize_clusters_1d(orig_x, orig_y, ms_indices, title)
    elif cluster_name == clustering_algos[6]:
        if pca_dpm:
            trajectory_xy,_ = get_pca_components(trajectory_xy)
        (aic, bic), dp_indices = dirichlet_process_clustering(trajectory_xy, dp_components[1], a=alpha[1], n_iter=n_iter)
        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(str(alpha[1]).replace(".", ""), dp_components[1])
        visualize_clusters_1d(orig_x, orig_y, dp_indices, title)
        """(aic, bic), dp_indices = dirichlet_process_clustering(trajectory_features, dp_components[1], a=alpha[1], n_iter=n_iter)
        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(alpha[1], dp_components[1])
        visualize_clusters_1d(fit_x, fit_y, dp_indices, title)"""
    elif cluster_name == clustering_algos[7]:
        hmm_indices, hmm_states = get_hmm_states(trajectory_xy, n_comp=hmm_comp[1])
        title = 'HMM_Model_component={}'.format(hmm_comp[1])
        visualize_clusters_1d(orig_x, orig_y, hmm_indices, title)
    elif cluster_name == clustering_algos[8]:
        aic_list = cPickle.load(open('aic_pickle.p', 'rb'))
        bic_list = cPickle.load(open('bic_pickle.p', 'rb'))
        cv_types=['spherical', 'tied', 'diag', 'full']
        n_components=[10,15, 20, 21, 22, 23, 24, 25]
        alphas=[.1, 1, 10, 100]
        visualize_model_selection(bic_list, alphas, n_components, cv_types)
        visualize_model_selection(aic_list, alphas, n_components, cv_types, model='AIC')
    elif cluster_name == clustering_algos[9]:
        k_range = range(15,35)
        mse_list_k_means = cPickle.load(open('mse_pickle_k_means.p','rb'))
        best_k_means = k_range[mse_list_k_means.index(min(mse_list_k_means))]
        plt.figure()
        plt.bar(k_range, mse_list_k_means)
        plt.title('Variance Average of Clusters for Different K Values K-MEANS')
        alphas = [0.01, 0.1, 1, 10, 100, 1000]
        dpm_list = cPickle.load(open('mse_pickle_dpm_with_alpha_001_01_1_10_100_1000.p','rb'))
        best_alpha = alphas[dpm_list.index(min(dpm_list))]
        plt.figure()
        plt.title("Variance Average of Clusters for Different Alpha Values DPM")
        plt.bar(alphas, dpm_list)
        dp_range = range(15, 35)
        plt.figure()
        dp_with_different_k_values = cPickle.load(open('mse_pickle_dpm_different_upper_bounds.p', 'rb'))
        best_dp_k_for_alpha_equals_10 = dp_range[dp_with_different_k_values.index(min(dp_with_different_k_values))]
        plt.title("Variance Average of Clusters for Different Upper Bound Values DPM, alpha=10")
        plt.bar(dp_range, dp_with_different_k_values)
        plt.show()
    elif cluster_name == clustering_algos[10]:
        dp_intra, dp_inter, dp_dbi = evaluation_parameters_of_clustering_method(feature_matrix, dp_indices)
        dp_inter_sum = sum(dp_inter.values())
        k_intra, k_inter, k_dbi = evaluation_parameters_of_clustering_method(feature_matrix, k_indices)
        k_inter_sum = sum(k_inter.values())
        ms_intra, ms_inter, ms_dbi = evaluation_parameters_of_clustering_method(feature_matrix, ms_indices)
        ms_inter_sum = sum(ms_inter.values())
        # Visualization Constants
        ind = np.arange(3)
        width = 0.35
        # Intra Variance Clusters
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects = ax.bar(ind, (dp_intra, k_intra, ms_intra), width, color='cmy')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        ax.set_title('Intra cluster variance for Clustering Algorithms')
        ax.set_xticks(ind+ 0.5*width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        fig.savefig('intra_cluster_variance_feature_matrix.pdf')
        fig.show()
        # Inter Variance Clusters
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects = ax.bar(ind, (dp_inter_sum, k_inter_sum, ms_inter_sum), width, color='cmy')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        ax.set_title('Inter cluster variance for Clustering Algorithms')
        ax.set_xticks(ind+ 0.5*width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        fig.savefig('inter_cluster_variance_feature_matrix.pdf')
        fig.show()
        # Davies Bouldin Index
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects = ax.bar(ind, (dp_dbi, k_dbi, ms_dbi), width, color='cmy')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        ax.set_title('Davies Bouldin Index for Clustering Algorithms')
        ax.set_xticks(ind+ 0.5*width)
        ax.set_xticklabels(('Dirichlet Process Mixture', 'K-Means', 'Mean Shift'))
        fig.savefig('davies_bouldin_index_feature_matrix.pdf')
        fig.show()
    elif cluster_name == clustering_algos[11]:
        # Deciding K values
        k_range = np.arange(1,100)
        k_dbi_list = list()
        for k in k_range:
            k_indices = k_means_clustering(feature_matrix, n_clusters=n_clusters[0])
            _, _, k_dbi = evaluation_parameters_of_clustering_method(feature_matrix, k_indices)
            k_dbi_list.append(k_dbi)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects = ax.bar(k_range, k_dbi_list, width, color='cmyrgb')
        #ax.set_xticks(ind+width)
        ax.set_title('Davies Bouldin Index for Different K Values')
        #ax.set_xticks(ind+ 0.5*width)
        fig.savefig('davies_bouldin_index_for_different_k_values.pdf')
        fig.show()
    elif cluster_name == clustering_algos[12]:
        k_indices = k_means_clustering(data_feature, n_clusters=n_clusters[0])
        title = 'K-Means_Clustering_with_k={}'.format(n_clusters[0])
        ms_indices, ms_labels, ms_n_clusters = mean_shift_clustering(data_feature)
        title = 'Mean-Shift_Clustering_with_ms_n_clusters={}'.format(ms_n_clusters)
        (aic, bic), dp_indices = dirichlet_process_clustering(data_feature, dp_components[0], a=alpha[0], n_iter=n_iter)
        title = 'Dirichlet_Process_Mixture_Clustering_with_alpha={}_upper_bound={}'.format(str(alpha[0]).replace(".", ""), dp_components[0])
        k_list, k_size_dict = build_cluster_index_signal(k_indices)
        ms_list, ms_size_dict = build_cluster_index_signal(ms_indices)
        dp_list, dp_size_dict = build_cluster_index_signal(dp_indices)
        k_size = [k_size_dict[ii] for ii in k_list]
        ms_size = [float(ms_size_dict[ii])/2 for ii in ms_list]
        dp_size = [dp_size_dict[ii] for ii in dp_list]
        x = range(len(dp_list))
        avg_signal = get_average_signal(delay)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x,k_list, s=k_size, c='m')
        ax.set_title('K-Means Clustering Indices for Spike Segments')
        fig.savefig('k_means_cluster_index_for_spike_segments.pdf')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, ms_list, s=ms_size, c='y')
        ax.set_title('Mean Shift Clustering Indices for Spike Segments')
        fig.savefig('mean_shift_cluster_index_for_spike_segments.pdf')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x,dp_list, s=dp_size, c='c')
        ax.set_title('DPM Clustering Indices for Spike Segments')
        fig.savefig('dpm_cluster_index_for_spike_segments.pdf')
        fig, ax = plt.subplots()
        avg_colors = plt.cm.Paired(np.linspace(0,1,len(avg_signal)))
        #ax.scatter( range(len(avg_signal)), [0]*len(avg_signal), color=colors )
        ax.scatter(range(len(avg_signal)), avg_signal, color=avg_colors)
        ax.set_title('Average Signal Colored with Time')
        k_colors = plt.cm.Paired(np.linspace(0,1, max(k_list)+1))
        ms_colors = plt.cm.Paired(np.linspace(0,1, max(ms_list)+1))
        dp_colors = plt.cm.Paired(np.linspace(0,1, max(dp_list)+1))
        k_color_list = [k_colors[ii] for ii in k_list]
        ms_color_list = [ms_colors[ii] for ii in ms_list]
        dp_color_list = [dp_colors[ii] for ii in dp_list]
        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_signal)), avg_signal, color=k_color_list)
        ax.set_title('K-Means Colored Spike Segments')
        fig.savefig('k_means_colored_spike_segments.pdf')
        fig, ax = plt.subplots()
        ax.scatter(range(len(k_colors)), [0]*len(k_colors), color=k_colors)
        fig.savefig("k_means_colors_cluster_index.pdf",bbox_inches='tight')
        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_signal)), avg_signal, color=ms_color_list)
        ax.set_title('Mean Shift Colored Spike Segments')
        fig.savefig('mean_shift_colored_spike_segments.pdf')
        fig, ax = plt.subplots()
        ax.scatter(range(len(ms_colors)), [0]*len(ms_colors), color=ms_colors)
        fig.savefig("ms_colors_cluster_index.pdf",bbox_inches='tight')
        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_signal)), avg_signal, color=dp_color_list)
        ax.set_title('DPM Colored Spike Segments')
        fig.savefig('dpm_cluster_colored_spike_segments.pdf')
        fig, ax = plt.subplots()
        ax.scatter(range(len(dp_colors)), [0]*len(dp_colors), color=dp_colors)
        fig.savefig("dpm_colors_cluster_index.pdf",bbox_inches='tight')

        spike_data, length_spikes, starting, ending = dmc.load_spike_data_3d()
        avg_time_signal = get_average_signal(spike_data)

        k_color_dict = dict()
        k_index_dict = dict()
        for k,v in length_spikes.iteritems():
            k_color_dict[k] = [k_color_list[k]] * v
            k_index_dict[k] = [k_list[k]] * v
        k_time_color = list(itertools.chain(*k_color_dict.values()))
        k_time_index = list(itertools.chain(*k_index_dict.values()))
        k_ind = {}
        for cluster in set(k_list):
            k_ind[cluster] = [i for i, x in enumerate(k_time_index) if x == cluster]

        ms_color_dict = dict()
        ms_index_dict = dict()
        for k,v in length_spikes.iteritems():
            ms_color_dict[k] = [ms_color_list[k]] * v
            ms_index_dict[k] = [ms_list[k]] * v
        ms_time_color = list(itertools.chain(*ms_color_dict.values()))
        ms_time_index = list(itertools.chain(*ms_index_dict.values()))
        ms_ind = {}
        for cluster in set(ms_list):
            ms_ind[cluster] = [i for i, x in enumerate(ms_time_index) if x == cluster]

        dp_color_dict = dict()
        dp_index_dict = dict()
        for k,v in length_spikes.iteritems():
            dp_color_dict[k] = [dp_color_list[k]] * v
            dp_index_dict[k] = [dp_list[k]] * v
        dp_time_color = list(itertools.chain(*dp_color_dict.values()))
        dp_time_index = list(itertools.chain(*dp_index_dict.values()))
        dp_ind = {}
        for cluster in set(dp_list):
            dp_ind[cluster] = [i for i, x in enumerate(dp_time_index) if x == cluster]

        beginning_range = 100
        ending_range = 500

        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_time_signal[beginning_range:ending_range])),
                                        avg_time_signal[beginning_range:ending_range],
                                        color=k_time_color[beginning_range:ending_range])
        ax.set_title('K-Means Colored Averaged Time Signal')

        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_time_signal[beginning_range:ending_range])),
                                        avg_time_signal[beginning_range:ending_range],
                                        color=ms_time_color)
        ax.set_title('Mean Shift Colored Averaged Time Signal')

        fig, ax = plt.subplots()
        ax.scatter(range(len(avg_time_signal[beginning_range:ending_range])),
                                       avg_time_signal[beginning_range:ending_range],
                                        color=dp_time_color[beginning_range:ending_range])

        ax.set_title('Dirichlet Process Colored Averaged Time Signal')

        visualize_average_time_signal_clusters(avg_time_signal,
                                                                        k_ind,
                                                                        save=True, title='K-Means Time Signal')

        visualize_one_sample_from_cluster_delay_map(delay, k_indices, title='K-Means Delay Map')

        visualize_average_time_signal_clusters(avg_time_signal,
                                                                        ms_ind,
                                                                        save=True, title='Mean-Shift Time Signal')
        visualize_one_sample_from_cluster_delay_map(delay, ms_indices, title='Mean-Shift Delay Map')

        visualize_average_time_signal_clusters(avg_time_signal,
                                                                        dp_ind,
                                                                        save=True, title='DPM Time Signal')
        visualize_one_sample_from_cluster_delay_map(delay, dp_indices, title='DPM Delay Map')


        """
        stream = list()
        stream.append(k_list)
        stream.append(ms_list)
        stream.append(dp_list)
        import stream_graph as sg
        sg.stacked_graph(stream)
        """