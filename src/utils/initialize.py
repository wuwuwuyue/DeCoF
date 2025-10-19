from glob import glob
import os
from glob import glob 
import os
import sys
import numpy as np
from PIL import Image
import random
import warnings
import json
import os.path as osp



def init_ff(phase,data_name='text2video_zero'):
	root = ''

	dataset_path_r=os.path.join(root,phase,data_name,'0_real/')
	
	dataset_path_f=os.path.join(root,phase,data_name,'1_fake/')
	



	
	real_img_list   = sorted(glob(dataset_path_r+'*'))
	fake_img_list  = sorted(glob(dataset_path_f+'*'))
	



				
	fake_label_list = [1 for _ in range(len(fake_img_list))]
	print(fake_img_list[0])

	
		
	real_label_list = [0 for _ in range(len(real_img_list))]
	
	img = real_img_list+fake_img_list
	label = real_label_list+fake_label_list

	return img,label


	

	
