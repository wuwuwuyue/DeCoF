import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from PIL import Image
import random
import warnings
from initialize import *
warnings.filterwarnings('ignore')


class Dataset(Dataset):
	def __init__(self,phase='train',data_name='text2video_zero',image_size=224,n_frames=8):
		print('1')
		assert phase in ['train','val','test']
		
		folder_list,label_list=init_ff(phase,data_name,n_frames=n_frames)
		
		print(f'SBI({phase}): {len(folder_list)}')
	

		self.folder_list=folder_list
		self.label_list=label_list
		self.image_size=(image_size,image_size)
		self.image_size_w=image_size
		self.phase=phase
		self.n_frames=n_frames

		#self.transforms=self.get_transforms()
		self.transforms=self.binary_dataset(self.phase)
	
        

	def __len__(self):
		return len(self.folder_list)

	def __getitem__(self,idx):
		data={}
		label=torch.tensor(self.label_list[idx])
		images_temp_r=sorted(glob(self.folder_list[idx]+'/*.jpg'))
		real_images=torch.zeros([self.n_frames,3,self.image_size_w,self.image_size_w],dtype=torch.float)
		for idx_x in range(self.n_frames):
			filename_r=images_temp_r[idx_x]
			img_r=Image.open(filename_r).convert("RGB")

			img_r=self.transforms(img_r)
		
		
			real_images[idx_x]=img_r
			
		
		data['img']=real_images
		data['label']=label
		return data
			
	def binary_dataset(self,phase):
		if phase=='train':
			crop_func = transforms.RandomCrop(224)
		
		else:
			crop_func = transforms.CenterCrop(224)

		if  phase=='train' :
			flip_func = transforms.RandomHorizontalFlip()
			aug=transforms.Lambda(lambda img: self.data_augment(img))
		else:
			flip_func = transforms.Lambda(lambda img: img)
			aug=transforms.Lambda(lambda img: img)
		if phase!='train' :
			rz_func = transforms.Resize((224,224))
		else:
			# rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
			rz_func = transforms.Resize((256,256))

		dset =transforms.Compose([
					rz_func,
					aug,
					crop_func,
					flip_func,
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				])
		return dset
	
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	from initialize import *
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=Dataset(phase='train',image_size=224,n_frames=8)
	batch_size=4
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					num_workers=20,
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	print(data['label'])
	i=0
	img=data['img']
	print(img.shape)
	img=img.permute(0,2, 1, 3, 4).contiguous().view((-1,3,224,224))
	utils.save_image(img, '/home/lma/Text2Video-Zero-main/ours/loader/loader_{}.jpg'.format(str(i)), nrow=8, normalize=False, value_range=(0, 1))
	data=next(data_iter)
	label=data['label']	


