import scipy.io
import os
import numpy as np

def load_raw_data_seg(obj = '/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/raw_data_40/data_40_1'):
    Video_inter = scipy.io.loadmat(obj)
    Video = Video_inter['VideoData']
    return (Video)

def load_data_40_obj(obj = '/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/obj_data_40'):
    obj_inter = scipy.io.loadmat(obj)
    obj = obj_inter['obj']
    return (obj)

def load_delay_maps(delay_map_str='/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/delay_map.mat'):
    """ Loads the delay maps from the dictionary which has different
    reference signals(average and max). Moreover, it has two options for the
    amplitudes of the signals if it is normalized or not => there are four
    different delay maps:
    Average - normalized
    Average - not normalized
    Max - normalized
    Max - not normalized
    """
    delay_map_dict = scipy.io.loadmat(delay_map_str)
    delay_map_average_normalized = delay_map_dict['average_normalized']
    delay_map_max_normalized = delay_map_dict['max_normalized']
    delay_map_average = delay_map_dict['average']
    delay_map_max = delay_map_dict['max']

    return (delay_map_average_normalized, delay_map_max_normalized,
    delay_map_average, delay_map_max)

def load_data_40_maps(delay_map_str='/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/data_40_delay.mat'):
    feat_dict = scipy.io.loadmat(delay_map_str)
    delay = feat_dict['delay']
    energy = feat_dict['energy']
    mask = feat_dict['mask']    
    return (delay, energy,mask)

    
def load_raw_correlation(delay_map_str ='/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/all_correlation.mat'):
    all_correlation = scipy.io.loadmat(delay_map_str)
    raw_correlation = all_correlation['intra_corr']
    return(raw_correlation)

def load_2d_delay_maps(delay_map_average_normalized,
                                       delay_map_max_normalized,
                                       delay_map_average,
                                       delay_map_max):
    dim1 = delay_map_average_normalized.shape[0]
    dim2 = delay_map_average_normalized.shape[1]
    dim3 = delay_map_average_normalized.shape[2]
    delay_map_a_n = np.reshape(delay_map_average_normalized,
        (dim1*dim2, dim3)).T
    delay_map_m_n = np.reshape(delay_map_max_normalized,
        (dim1*dim2, dim3)).T
    delay_map_a = np.reshape(delay_map_average,
        (dim1*dim2, dim3)).T
    delay_map_m = np.reshape(delay_map_max,
        (dim1*dim2, dim3)).T
    return delay_map_a_n, delay_map_m_n, delay_map_a, delay_map_m,

def load_features_and_delay_map(data_str='/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/feature_data/features_and_delay_map.mat'):
    feat_dict = scipy.io.loadmat(data_str)
    delay = feat_dict['delay']
    energy = feat_dict['energy']
    mask = feat_dict['mask']
    after_pca = feat_dict['score']
    eigen = feat_dict['latent']
    starting, ending, indices = load_start_end_frame_numbers()
    return (delay, energy, mask, after_pca, eigen, starting, ending, indices)

def load_start_end_frame_numbers(data_folder='data',
                        data_str='start_end_frame_numbers.mat'):
    data_path = os.path.join(data_folder, data_str)
    start_end_dict = scipy.io.loadmat(data_path)
    start_frames = start_end_dict['start']
    end_frames = start_end_dict['endd']
    indices = start_end_dict['clustering_indices']
    return start_frames, end_frames, indices

def load_spike_data_3d(data_folder='data', data_str='data_3d.mat'):
    data_path = os.path.join(data_folder, data_str)
    data_3d_dict = scipy.io.loadmat(data_path)
    data_3d = data_3d_dict['data_3d']
    starting, ending, indices = load_start_end_frame_numbers()
    starting = starting.tolist()[0]
    ending = ending.tolist()[0]
    spike_data = np.zeros((18,20,0))
    length_spikes = dict()
    for ii, (s,e) in enumerate(zip(starting,ending)):
        temp = data_3d[:,:,s:e]
        spike_data = np.concatenate((spike_data, temp), axis=2)
        length_spikes[ii] = e-s
    return spike_data, length_spikes, starting, ending

def load_centroids_of_data(data_folder='data', data_str='centroid_clusters.mat'):
    data_path = os.path.join(data_folder, data_str)
    data_dict = scipy.io.loadmat(data_path)
    orix = data_dict['orix']
    oriy = data_dict['oriy']
    fitx = data_dict['fitx']
    fity = data_dict['fity']
    orig_x = dict()
    orig_y = dict()
    fit_x = np.zeros((fitx.size, fitx[0,0].shape[1]))
    fit_y = np.zeros((fity.size, fity[0,0].shape[1]))

    for ii in range(fitx.size):
        fit_x[ii,:] = fitx[0,ii]

    for ii in range(fity.size):
        fit_y[ii,:] = fity[0,ii]

    for ii in range(orix.size):
        orig_x[ii] = orix[0,ii][:,0]

    for ii in range(oriy.size):
        orig_y[ii] = oriy[0,ii][:,0]

    return (fit_x, fit_y, orig_x, orig_y)

def load_resampled_trajectories(data_folder='data', data_str='cluster_10_20.mat'):
    data_path = os.path.join(data_folder, data_str)
    data_dict = scipy.io.loadmat(data_path)
    x = data_dict['cluster_10_20_x']
    y = data_dict['cluster_10_20_y']
    label = data_dict['cluster_10_20_label'][0]
    return x,y,label

def load_delay_energy_binary_map(data_folder='data', data_str='delay_energy_binary_map.mat'):
    data_path = os.path.join(data_folder, data_str)
    data_dict = scipy.io.loadmat(data_path)
    delay = data_dict['delay']
    energy = data_dict['energy']
    binary = data_dict['mask']
    return delay, energy, binary

def load_indices(data_folder='data', data_str='indices.mat'):
    data_path = os.path.join(data_folder, data_str)
    data_dict = scipy.io.loadmat(data_path)
    indices = data_dict['indices']
    return indices




if __name__ == '__main__':
    delay, energy, mask, after_pca, eigen, starting, ending, indices = load_features_and_delay_map()