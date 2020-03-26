# Anomaly Detection in Video Sequence with Appearance-Motion Correspondence



## Abstract

-  Learning a correspondence between **common object appearances** (e.g. pedestrian, background, tree, etc.) and **their associated motions**.
- A combination of a reconstruction network and an image translation model that share the same encoder.
  - **a reconstruction network** :  the most significant structures that appear in video frames
  - **an image translation model**  :  associate motion templates to such structures.
- The training stage is performed using **only videos of normal events** and the model is then capable to estimate frame-level scores for an unknown input.

<div style="page-break-after: always; break-after: page;"></div>

## Introduction

- ### Contribution

  - We design a CNN that combines a Conv-AE and a UNet, in which each stream has its own contribution for the task of detecting anomalous frames. The model can be trained end-to-end.
  - We integrate an Inception module modified from [48] right after the input layer to reduce the effect of network’s depth since this depth is considered as a hyper parameter that requires a careful selection.
  - We propose a patch-based scheme estimating frame level normality score that reduces the effect of noise which appears in the model outputs.

<div style="page-break-after: always; break-after: page;"></div>

## Proposed Method

![11](11.PNG)





- Conv-AE
  - common appearance spatial structure in normal events
- Motion Decoder
  - To determine an association between each input pattern and its corresponding motion represented by an optical flow of 3 channels (xy displacements and magnitude).
  
- The skip connections in U-Net.
  - Motion Decoder : for image translation since it directly transforms low-level features (e.g. edge, image patch) from original domains to the decoded ones.



- Not use any fully-connected layer
  - 이론적으로는 해상도에 상관없음.
  - 모델을 단순화 하기 위해 사이즈 input size 고정:  128 × 192 × 3
  - 1:1.5 비율 사용 (?)

<div style="page-break-after: always; break-after: page;"></div>

### Inception module



- To let the model select its appropriate convolutional operations.
- Remove the max pooling in this module since the input is a regular video frame instead of a collection of feature maps. 
- filter sizes 1 × 1, 3 × 3, 5 × 5 and 7 × 7



### Appearance convolutional autoencoder

-  the detection of strange (abnormal) objects within input frames by learning common appearance templates in normal events.

- Encoder

  - convolution, batch-normalization (BatchNorm) and leakyReLU activation 
  - The first block (right after the Inception module) does not contain BatchNorm layer as suggested in [17] for our U-Net task 
  - Instead of using pooling layer to reduce the resolution of feature maps, we apply strided convolution
    - Such parametric operation is expected to support the network finding an informative way to downsample the spatial resolution of feature maps as well as learning the further upsampling in decoding stage

- Decoder

  - A dropout layer (with pdrop = 0.3) is attached before the ReLU activation in each block as a regularization that reduces the risk of overfitting during the training stage

- Loss

  - $l_2$ distance between the input image $I$ and its reconstruction $\hat{I}$ : 
    $$
    \mathcal{L}_{\text{int}}(I, \hat{I}) = ||I-\hat{I}||^2_2 \qquad \qquad \qquad \qquad  \qquad \qquad  \qquad \qquad (1)
    $$
    
- A drawback of using only $l_2$ loss is the blur in the output:
    $$
    \mathcal{L}_{\text{grad}} (I, \hat{I}) = sum_{d \in \{x,y\}} \left| \left| |g_d(I)| - |g_d(\hat{I})| \right| \right|_1 \qquad \qquad\ \qquad (2)
    $$
    
  
  
  
$$
  \mathcal{L}_{\text{appe}} (I, \hat{I}) = \mathcal{L}_{\text{int}} (I, \hat{I}) + \mathcal{L}_{\text{grad}} (I, \hat{I}) \qquad \qquad\ \qquad \qquad \qquad (3)
  $$
  
<div style="page-break-after: always; break-after: page;"></div>

### Motion prediction U-Net

Our UNet sub-network thus focuses on learning the association between such patterns and corresponding motions. 



ground truth optical flow([retrained FlowNet2) 사용



The decoder of our U-Net has the same structure as the Conv-AE except for the skip connections.



- Loss
  - $l_1$ distance
    -  the FlowNet2 model is formed as a fusion of multiple networks providing optical flows from coarse (noisy) to fine (smooth), the result might thus contain noise or even amplify noisy regions during the smoothing procedure.
    - because the selection of optical flow estimation is not limited to FlowNet2, the training ground truth obtained from other algorithms might therefore possibly have small patches of wrong and/or noisy motion measure




$$
\mathcal{L}_{\text{flow}}(F_t, \hat{F}_t) = ||F_t-\hat{F}_t||_1 \qquad \qquad \qquad \qquad  \qquad \qquad  \qquad \qquad (4)
$$


$F_t$ : the ground truth optical flow estimated from two consecutive frames $I_t$ and $I_{t+1}$ 

$\hat{F}_t$ : the output of our U-Net given $I_t$  

This stream attempts to predict instant motions of objects appearing in the video.



<div style="page-break-after: always; break-after: page;"></div>

###  Additional motion-related objective function





<div style="page-break-after: always; break-after: page;"></div>

###  Anomaly detection

<div style="page-break-after: always; break-after: page;"></div>

## Experiment

- CUHK Avenue
- UCSD Ped2
- Subway Entrance Gate and Exit Gate
- Traffic-Belleview와 Traffic-Train

![1](1.png)



- Ground truth optical flow estimator
  - FylingTihing3D와 ChairsSDHom 데이터 셋으로 학습시킨  FlowNet2



<div style="page-break-after: always; break-after: page;"></div>

### CUHK Avenue and UCSD Ped2

![2](2.png)

![3](3.png)

- 트럭은 처음 보는 객체이기 때문에 보행자의 패턴으로써 reconstruct 되었다.
- 그래서 트럭의 predicted motion은 ground truth와 완전히 다르다.
- 맨 오른쪽 자전거도 마찬가다.

![4](4.png)





- 자전거는 처음 보는 객체이기 때문에 보행자와 배경과 비슷한 밝기로 표시 되었다.

  ![5](5.png)





- 모델은 훈련 데이터에서 관찰된 것 처럼 느린 이동속도와 다른 모션 방향을 예상했다.
- 재구성한 남자의 바지 색상은 배경이 잘 복원된 상태에서 입력 프레임과 약간 다르다. 이것은 패턴의 색상과 움직임 사이에 낮은 유의관계를 보여준다.

<div style="page-break-after: always; break-after: page;"></div>

### Subway Entrance and Exit gates



- 비정상적인 이벤트
  - 잘못된 방향 (탑승자가 출입구를 통해서 나가는 경우)
  - 지불을 하지 않는 경우
  - 배회 (loitering)
  - 불규칙한 상호작용
    - 다른 사람을 피하기 위해 어색하게 걷는 사람
    - 갑자기 걷는 속도를 변경하는 경우

![6](6.png)





- FA(false alarm)이 다른 방법에 비해 높다. 
  - 테스트 세트에서 정상으로 표시된 일부 이벤트가 다른 상황에서는 이상으로 간주될 수 있기 때문.



#### false alarm과 missed anomaly detection의 visualization



![7](7.png)



- movement stopping과 loitering에 대한 normality decision이 불안정 하다.
  -  (a)-(e)는 miss 했고 (f)-(h)는 잘 못 detect 했다.





원인

- anomaly score가 smoothly 혹은 slowly 하게 변할 때  maxmium localization을 사용하는 것은 적절하지 않다.
- training set에 (b)와 (e)에 있는 남자가 loitering하는 데이터가 있다. 
- (h)는 loitering하는 남자가 오른쪽 사이드에 보이지만 anomaly로 label되지 않아 ground truth annotation이 애매하다.
- (i)  에서는 모델이 남자가 left gate로 들어 갈 것으로 예상했지만 갑자기 오른쪽 문으로 바뀌었다. 이런 행동은 훈련 데이터에 나타나지 않아 모델은 anomalous event로 예측했다.
- (j)는 motion stream 이 위치에있는 대부분의 사람들이 훈련 데이터에서 좌측으로 이동하기 때문에 승객이 기차에 갈 것을 예상했다. 



<div style="page-break-after: always; break-after: page;"></div>

### Traffic-Belleview and Traffic-Train



Traffic-Belleview

- 좌우로 움직이는 차를 이상으로 간주



Traffic-Traing

- camera jitter에 따라 lighting condition이 심하게 변한다.
- 사람의 움직임을 이상으로 간주

![9](9.png)



![8](8.png)



- 움직임이 매우 nosy 하고 가운데 승객을 error map에서는 놓쳤다. 



- 카메라 지터의 영향을 줄이기 위한 시도로, motion의 support 없는 다른 프레임 레벨 점수를 추정했다. 
- 구체적으로는 Structure Similarity Idex(SSIM)[50]를 사용하여 입력 프레임과  appearance stream이 만드는 reconstruction의 유사성을 계산했다. MSE나 PSNR과 같은 다른 일반적인 측정과 비교하여 SSIM은 픽셀별 비교가 적절하지 않은 지터 영상에서 잘 작동할 수 있다.
- 표 3은 이러한 변경이 특히 열차 데이터 세트를 통해 이상 징후 감지 결과를 개선했음을 보여준다. ROC 및 PR 곡선, 일부 형상 지도의 시각화 및 각 단일 스트림의 평가 결과를 포함한 자세한 내용은 보충 자료에서 제공된다.

![10](10.png)





- optical flow estimator의 영향은 두개의 차가 big blob로 결합된 Figure 6(c)에 잘 설명되어 있다. 이 잘못된 추정으로 인해 다른 방향으로 달리는 3대의 차는 정확하게 찾아 냈음에도 불구하고 error map에 큰 영향을 미쳤다.
  - 다른 optical flow 를 선택하거나 FlowNet2를 좀 더 적절한 데이터 셋으로 pretrain 시킴으로써 개선할 수 있을 것이다.





