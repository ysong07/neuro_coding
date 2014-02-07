import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
#import ecog_animation as ea
import delay_map_clustering as dmc

class visulizer:
    def __init__(self):
        
        obj = dmc.load_data_40_obj()
        data = obj['data']
        video_data = data[0][0]
        self.data = video_data
        self.start = obj['start']
        self.end = obj['end']
        
    def V_seg_num(self,n_model):
        n_model = n_model -1
        video = self.data[0,n_model]
        ims = []
        fig = plt.figure()
        for i in range(video.size/18/20):
            im = plt.imshow(video[:,:,i])
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=100)
        name_str =  str(n_model+1)+'.mkv'
        ani.save(name_str, writer=animation.FFMpegFileWriter(), metadata={'artist':'Guido'})

    def get_seg_t(self,n_model):
        return(int(self.start[0][0][:,n_model-1]),int(self.end[0][0][:,n_model-1]))

    def V_time(self,start_t,end_t,folder_path = '/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/raw_data_40/'):
        if (start_t//55400 == end_t//55400):
            file_str = folder_path + 'data_40_' + str(start_t//55400+1)
            VideoData = dmc.load_raw_data_seg(file_str)
            t_s = start_t%55400
            t_e = end_t%55400
        else:
            file_str = folder_path + 'data_40_' + str(start_t//55400+1)
            VideoData1 = dmc.load_raw_data_seg(file_str)
            file_str = folder_path + 'data_40_' + str(start_t//55400+2)
            VideoData2 = dmc.load_raw_data_seg(file_str)
            VideoData = np.concatenate((VideoData1,VideoData2),axis =2)
            t_s = start_t%55400
            t_e = end_t%55400+55400
        ims =[]
        fig = plt.figure()
                # resort the data find opimal threshold for t_max,t_min
        for i in range(t_s,t_e)
            
        for i in range(t_s,t_e):
            im = plt.imshow(VideoData[:,:,i])
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=100)
        name_str =  str(start_t)+'  '+str(end_t)+'.mkv'
        ani.save(name_str, writer=animation.FFMpegFileWriter(), metadata={'artist':'Guido'})











