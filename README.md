# Detecting AI-Generated Video via Frame Consistency
<div align="center">
    <a href='https://arxiv.org/abs/2402.02085'><img src='https://img.shields.io/badge/ArXiv-2402.02085-red'></a>
</div>

![Overview](pics/model-1_1.png)  

## News 🚀
**[2025.4.16]** Our paper has been accepted by ICME 2025. 🎉🎉🎉 

**[2025.4.20]** In order to make the code more readable, we plan to spend some time organizing the code. However, many people have sent emails hoping to open source it as soon as possible, so we have compiled an open source code in a short period of time. However, it has not been verified and the code cannot run. Currently, the training code should have no problems. If there are any problems, please let us know directly. 

**[2025.4.16]** The training code has been open-source.

**[2024/6/7]**  The prompts used to generate videos, the attribute partitioning involved in prompts, and the partitioning of the dataset  have been open-source. You can access it in the `/datas/` folder. Unfortunately, we are unable to directly provide real videos. You can download them from the original dataset based on the video_id of prompts.  

**[2024/6/16]**  All generated videos can be downloaded from [here](https://drive.google.com/drive/folders/1X4Gw4hkWfka42IaBQ6ImkDTGAeA9Wlk4?usp=drive_link), The emergence speed of video generation models far exceeds our imagination. If you expand a subset based on our dataset, we sincerely invite you to release the corresponding generated videos.

**[2024/7.1]** Thanks to [Kling](https://kling.kuaishou.com/). We have extended the DecoF dataset based on the video generation model **Kling**, and the test dataset for Kling will be made public soon.


## Abstract
> The increasing realism of AI-generated videos has raised potential security concerns, making it difficult for humans to distinguish them from the naked eye. Despite these concerns, limited research has been dedicated to detecting such videos effectively.
To this end, we propose an open-source AI-generated video detection dataset. Our dataset spans diverse objects, scenes, behaviors, and actions by organizing input prompts into independent dimensions. 
It also includes various generation models with different generative models, featuring popular commercial models such as OpenAI's Sora, Google's Veo, and Kwai's Kling.
Furthermore,  we propose a simple yet effective **De**tection model based on **Co**ncistency of **F**rame (DeCoF), which learns robust temporal artifacts across different generation methods.
Extensive experiments demonstrate the generality and efficacy of the proposed DeCoF in detecting AI-generated videos, including those from nowadays' mainstream commercial generators.

<p align="center">
<img src="pics/figure1_1.jpg" width=60%>
</p>

## Code
> The training code is in src.

## Setup

1. Clone this repository 
```bash
git clone https://github.com/wuwuwuyue/DeCoF
cd DeCoF
```
2. Prepare the dataset
Download datasets and place them in `./data/` folder.  Extract frames from the first 32 frames of the video, evenly select 8 frames, crop the frames into squares at the center, and store them as follows. The files for dividing the training set, testing set, and validation set are stored in  `datas/split`.
For example, download **Text2Video_Zero** and **Real**,:
```
└── data
    └── Text2Video_Zero
        ├── Train 
        ├── Val
        ├── Test
        │   └── 0_real
        |       └── -_hbPLsZvvo_19_25
        |           └── 000.jpg
        |           └── 001.jpg
        │   └── 1_fake
        |       └── -_hbPLsZvvo_19_25
        |           └── 000.jpg
        |           └── 001.jpg    
```
3. Install the necessary libraries
```bash
pip install torch torchvision
pip install tqdm  einops  numpy scikit-learn pillow ftfy regex  opencv-python  scipy  opencv-python

```
## Training
    Run the training:
    ```bash
    CUDA_VISIBLE_DEVICES=* python3 src/train.py \
    src/configs/base.json \
    -n DeCoF
    ```
## Citation
If you find our work useful for your research, please consider citing our paper
```bibtex
@article{ma2024decof,
  title={DeCoF: Generated Video Detection via Frame Consistency},
  author={Ma, Long and Zhang, Jiajia and Deng, Hongping and Zhang, Ningyu and Liao, Yong and Yu, Haiyang},
  journal={arXiv e-prints},
  pages={arXiv--2402},
  year={2024}
}
```
## 📧 Contact
If you have any questions, please contact us: longm@mail.ustc.edu.cn.
