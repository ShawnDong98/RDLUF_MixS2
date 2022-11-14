# RDLUF-MixS2 for CASSI


This repo is the implementation of paper "Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging"


# Abstract

To acquire a snapshot spectral image, coded aperture snapshot spectral imaging (CASSI) is proposed. A core problem of the CASSI system is to recover the reliable and fine underlying 3D spectral cube from the 2D measurement. By alternately solving a data subproblem and a prior subproblem, deep unfolding methods achieve good performance. However, in the data subproblem, the used sensing matrix is ill-suited for the real degradation process due to the device errors caused by phase aberration, distortion; in the prior subproblem,  it is important to design a suitable model to jointly exploit both spatial and spectral priors. In this paper, we propose a Residual Degradation Learning Unfolding Framework (RDLUF), which bridges the gap between the sensing matrix and the degradation process. Moreover, a MixS2 Transformer is designed via mixing priors across spectral and spatial to strengthen the spectral-spatial representation capability. Finally, plugging the MixS2 Transformer into the RDLUF leads to an end-to-end trainable and interpretable neural network RDLUF-MixS2. Experimental results establish the superior performance of the proposed method over existing ones.

# Comparison with State-of-the-art Methods

<div align=center>
<img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/performance.png" width = "350" height = "300" alt="">
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

Diagram of the MixS2 Transformer. (a) MixS2 Transformer adopts a U-shaped structure with block interactions. (b) The basic unit of the MixS2 Transformer, MixS2 block. (c) The structure of the spectral self-attention branch. (d) The structure of the lightweight inception branch. (e) The components of the gated-Dconv feed-forward network(GDFN)


# Usage 

