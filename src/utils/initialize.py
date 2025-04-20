from glob import glob
import os
from glob import glob 
import os



def init_ff(phase,data_name='text2video_zero',n_frames=8):
	dataset_path_r=os.path.join('/home/Text2Video-Zero-main/ours/data/',phase,data_name,'0_real/')
	print(dataset_path_r)
	folder_list_r = sorted(glob(dataset_path_r+'*'))
	#print(len(folder_list_r))
	dataset_path_f=os.path.join('/home/Text2Video-Zero-main/ours/data/',phase,data_name,'1_fake/')
	folder_list_f = sorted(glob(dataset_path_f+'*'))
	label_list=[0]*len(folder_list_r)+[1]*len(folder_list_f)

	return folder_list_r+folder_list_f,label_list
	

	
