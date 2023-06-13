# RDLUF-Mix $S^2$ for CASSI


This repo is the implementation of paper "Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging"

|                          *Scene 1*                           |                          *Scene 6*                           |                          *Scene 8*                           |                          *Scene 10*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene1.gif"  height=170 width=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene6.gif" width=170 height=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene8.gif" width=170 height=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene10.gif" width=170 height=170> |


## News

- **2023.03.16**: release pretrained weights in `simulation/test_code/checkpoints/` and `real/test_code/checkpoints/`

# Abstract

To acquire a snapshot spectral image, coded aperture snapshot spectral imaging (CASSI) is proposed. A core problem of the CASSI system is to recover the reliable and fine underlying 3D spectral cube from the 2D measurement. By alternately solving a data subproblem and a prior subproblem, deep unfolding methods achieve good performance. However, in the data subproblem, the used sensing matrix is ill-suited for the real degradation process due to the device errors caused by phase aberration, distortion; in the prior subproblem,  it is important to design a suitable model to jointly exploit both spatial and spectral priors. In this paper, we propose a Residual Degradation Learning Unfolding Framework (RDLUF), which bridges the gap between the sensing matrix and the degradation process. Moreover, a Mix $S^2$ Transformer is designed via mixing priors across spectral and spatial to strengthen the spectral-spatial representation capability. Finally, plugging the Mix $S^2$ Transformer into the RDLUF leads to an end-to-end trainable and interpretable neural network RDLUF-Mix $S^2$ . Experimental results establish the superior performance of the proposed method over existing ones.

# Comparison with state-of-the-art methods

<div align=center>
<img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/Teaser.png" width = "350" height = "300" alt="">
</div>

 PSNR-Parameters comparisons with previous HSI reconstruction methods. The vertical axis is PSNR (in dB performance), and the horizontal axis is Parameters(memory cost). Our proposed Residual Degradation Learning Unfolding Framework with Mixing priors across Spatial and Spectral(RDLUF-Mix $S^2$ ) Transformers outperforms previous methods while requiring fewer parameters.

# Architecture

## Residual Degradation Learning Unfolding Framework

<div align=center>
<img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/RDLUF.png" width = "700" height = "300" alt="">
</div>

The architecture of our RDLUF with $K$ stages (iterations). RDLGD and PM denote the Residual Degradation Learning Gradient Descent module and the Proximal Mapping module in each stage. There is a stage interaction between stages.

## Mixing priors across Spectral and Spatial Transformer(PM)

<div align=center>
<img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/MixS2T.png" width = "700" height = "500" alt="">
</div>

Diagram of the Mix $S^2$ Transformer. (a) Mix $S^2$ Transformer adopts a U-shaped structure with block interactions. (b) The basic unit of the MixS2 Transformer, Mix $S^2$ block. (c) The structure of the spectral self-attention branch. (d) The structure of the lightweight inception branch. (e) The components of the gated-Dconv feed-forward network(GDFN)



# Usage 

## Prepare Dataset:

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--RDLUF_MixS2
    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--visualization
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask_3d_shift.mat
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask_3d_shift.mat
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

 We use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set.

## Simulation Experiement:

### Training


```
cd RDLUF_MixS2/simulation/train_code/

# RdLUF-MixS2 3stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 0  --clip_grad

# RdLUF-MixS2 5stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_5stage/ --method duf_mixs2 --stage 5 --body_share_params 1  --clip_grad

# RdLUF-MixS2 7stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_7stage/ --method duf_mixs2 --stage 7 --body_share_params 1  --clip_grad

# RdLUF-MixS2 9stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_9stage/ --method duf_mixs2 --stage 9 --body_share_params 1  --clip_grad
```

The training log, trained model, and reconstrcuted HSI will be available in `RDLUF_MixS2/simulation/train_code/exp/` .

### Testing

Place the pretrained model to `RDLUF_MixS2/simulation/test_code/checkpoints/`

Run the following command to test the model on the simulation dataset.

```
cd RDLUF_MixS2/simulation/test_code/


# RdLUF-MixS2 3stage
python test.py --template duf_mixs2 --stage 3 --body_share_params 0 --outf ./exp/duf_mixs2_3stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_3stage.pth

# RdLUF-MixS2 5stage
python test.py --template duf_mixs2 --stage 5 --body_share_params 1 --outf ./exp/duf_mixs2_5stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_5stage.pth

# RdLUF-MixS2 7stage
python test.py --template duf_mixs2 --stage 7 --body_share_params 1 --outf ./exp/duf_mixs2_7stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_7stage.pth

# RdLUF-MixS2 9stage
python test.py --template duf_mixs2 --stage 9 --body_share_params 1 --outf ./exp/duf_mixs2_9stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_9stage.pth
```

- The reconstrcuted HSIs will be output into `RDLUF_MixS2/simulation/test_code/exp/`
- Place the reconstructed results into `RDLUF_MixS2/simulation/test_code/Quality_Metrics/results` and

```
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.

### Visualization

- Put the reconstruted HSI in `RDLUF_MixS2/visualization/simulation_results/results` and rename it as method.mat, e.g., RDLUF_MixS2_9stage.mat
- Generate the RGB images of the reconstructed HSIs

```
cd RDLUF_MixS2/visualization/
Run show_simulation.m 
```

## Real Experiement:

### Training

```
cd RDLUF_MixS2/real/train_code/

# RDLUF-MixS2 3stage
python train.py --template duf_mixs2 --outf ./exp/rdluf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 1
```

The training log and trained model will be available in `RDLUF_MixS2/real/train_code/exp/`

### Testing

```
cd RDLUF_MixS2/real/test_code/

# RDLUF-MixS2 3stage
python test.py --template duf_mixs2 --outf ./exp/rdluf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 1 --pretrained_model_path ./checkpoints/RDLUF_MixS2_3stage.pth
```

The reconstrcuted HSI will be output into `RDLUF_MixS2/real/test_code/Results/`

### Visualization

- Put the reconstruted HSI in `RDLUF_MixS2/visualization/real_results/results` and rename it as method.mat, e.g., RDLUF_MixS2_3stage.mat.
- Generate the RGB images of the reconstructed HSI

```
cd RDLUF_MixS2/visualization/
Run show_real.m
```

## Acknowledgements

Our code is heavily borrowed from [MST](https://github.com/caiyuanhao1998/MST)  and [DGSMP](https://github.com/TaoHuang95/DGSMP), thanks for their generous open source.


## Citation

If this code helps you, please consider citing our works:

```shell
@inproceedings{dong2023residual,
  title={Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging},
  author={Dong, Yubo and Gao, Dahua and Qiu, Tian and Li, Yuyan and Yang, Minxi and Shi, Guangming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22262--22271},
  year={2023}
}
```
