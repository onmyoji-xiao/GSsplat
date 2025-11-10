<div align="center">

# GSsplat: Generalizable Semantic Gaussian Splatting for Novel-view Synthesis in 3D Scenes
**Feng Xiao** · **Hongbin Xu** · **Wanlin Liang** · **Wenxiong Kang**  
*South China University of Technology*  

[Code](https://github.com/onmyoji-xiao/GSsplat) | [Paper](https://arxiv.org/abs/2505.04659) | [Project Page](#)  

![image](overall.png)
</div>

## Environment
```
git clone https://github.com/onmyoji-xiao/GSsplat.git
cd Gssplat/

conda create -n gssplat python=3.10
conda activate gssplat

pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt

cd submodules/diff-gaussian-rasterization
python setup.py install
```

## Dataset
The dataset we use is the same as GSNeRF, both being ScanNet and Replica.   
For download instructions, please refer to the [GSNeRF](https://github.com/TimChou-ntu/GSNeRF) repository.

## Training
### Pre-trained Depth Estimation Model
First, obtain the pre-trained depth estimation model. Execute the following command
```
python depth_train.py --cfg ./configs/scannet_depth.yaml --save_dir ./save --gpus [0]
```
or directly download the [CasMVSNet model](https://drive.google.com/drive/folders/14QsAmHbixd9V53xzFkqwz05miRFdM2mF?usp=sharing) we have trained, including versions with different numbers of views.
