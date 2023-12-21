# STLR-DP
Code of Spectral-Temporal Low-Rank Regularization with Deep Prior for Thick Cloud Removal

## Running environment :
python=3.9, pytorch-gpu=1.12.1.

## To test: 
   Run the "main.py" to generate the thick cloud removal results.

## Architecture of STLR-DP

![alt text](https://github.com/zhentao-zou/STLR-DP/blob/main/Figure/Framework.png)
The framework of our proposed STLR method. (a) The cloud-contaminated images $\boldsymbol{\mathcal{Y}}$ $\in$ $\mathbb{R}^{m \times n \times bt}$ ($m$, $n$, $b$, and $t$ represent the height, width, spectral, and temporal dimensions, respectively.). Given the observed image $\boldsymbol{\mathcal{\bar{Y}}}$ $\in$ $\mathbb{R}^{m \times n \times b\times t}$, we reshape it into a third-order tensor $\boldsymbol{\mathcal{Y}}$ $\in$ $\mathbb{R}^{m \times n \times bt}$ as the algorithm input. (b) The tensor nuclear norm based on the tensor tubal rank is utilized to characterize the low-rank property of spatial factors $\boldsymbol{\mathcal{B}}$. (c) The orthogonal constraint on the spectral factors $\boldsymbol{Q}$. (d) We fine the network parameters in a self-supervised manner, which means that the model can work well using the single cloud-contaminated image solely without any extra external training data or pre-trained models. blue. The untrained neural network includes three types of modules, i.e., upsampling, downsampling, and skip-connections module. Each downsampling module consists of convolution, downsample, Batchnorm, LeaklyRelu, convolution, Batchnorm, and LeaklyRelu layer in sequence. Each upsampling module consists of Batchnorm, convolution, Batchnorm, LeaklyRelu, convolution, batchnorm, LeaklyRelu, and upsampling layer in sequence. Each skip-connection layer consists of convolution, batchnorm, and LeaklyRelu layer in sequence. (e) The variables in the model are updated by the ADMM algorithm.

## Reference
@article{zou2023spectral,  
  title={Spectral-Temporal Low-Rank Regularization with Deep Prior for Thick Cloud Removal},  
  author={Zou, Zhentao and Chen, Lin and Jiang, Xue},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  year={2024},  
  publisher={IEEE}  
}
