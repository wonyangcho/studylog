# SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY



- 목적
  - stable
  - sample efficient
    
- 논문에서 제안한 여러가지 방법들
  - **truncated importance sampling with bias correction**
  - **stochastic dueling network architectures**
  - **a new trust region policy optimization method**

- Replay Buffer
  - DQN에서 처음 적용
  - sample correlation을 줄이기 위한 용도로 사용된 기술이지만, 실제로는 **sample efficiency**도 향상 시킨다



## Background and Prbolem Setup



Max. $R_t = \mathbb{E}(\sum_{i \ge 0} \gamma^i r_{r+i})$ 

$Q^{\pi} (x_t, a_t) = \mathbb{E}_{x_{t+1}:\infty}, a_{t+1:\infty} [R_t |x_t,a_t] \qquad \qquad V^\pi(x_t) = \mathbb{E}_{a_t}[Q^\pi (x_t,a_t)|x_t]$

$A^\pi (x_t,a_t) = Q^\pi (x_t,a_t)-V^\pi (x_t) \qquad \qquad \qquad \mathbb{E}_{a_t}[A^\pi (x_t,a_t)] = 0$ 



$g = \mathbb{E}_{x_0:\infty, a_0:\infty} [ \sum_{t \ge 0} A^\pi(x_t,a_t) \bigtriangledown_\theta (a_t|x_t)] \qquad \qquad (1)$



**A3C**

- trade-off bias and variance 


$$
\hat{g}^{\text{a3c}} = \sum_{t \ge 0} \left(\left(\sum_{i=0}^{k-1} \gamma^i r_{t+i})+\gamma^k V^{\pi}_{\theta_v}(x_{t+k})-V^{\pi}_{\theta_v}(x_{t}) \right)\right) \bigtriangledown_\theta log \pi_\theta (a_t|x_t) \qquad \qquad (2)
$$


**ACER** 

- A3C + serveral modification , new modules
- a single deep neural network to estimate the policy $\pi_\theta (a_t | x_t)$ and value function  $V^{\pi}_{\theta_v} (x_t)$



### Discrete Actor Critic With Experience Replay



**off-policy learning with experience replay**

- off-policy learning with experience replay은 actor-critics의 **sample efficiency 를 향상** 시킨다. 
- 그러나 **off-policy의 variance와 stability를 conotrol하는 것은 어려운 일**이다. 
- **Importance sampling**은 off-policy learning의 가장 popular한 approach이다.


$$
\hat{g}^{\text{imp}} = \left( \prod_{t=0}^K \rho_t \right) \sum^k_{t=0} \left( \sum^k_{i=0} \gamma^i r_{t+i}\right) \bigtriangledown_\theta log \pi_\theta (a_t|x_t) \qquad \qquad (3) \\ \rho_t = {\pi(a_t|x_t) \over {\mu (a_t|x_t)}} \qquad \qquad \qquad \qquad  \qquad \qquad \qquad \qquad \qquad \qquad
$$

- $\left( \prod_{t=0}^K \rho_t \right)$ 은 **high variance**를 야기한다.



**marginal value functions over the limiting distribution ** of the process to yield the following approximation of the gradient:


$$
g^{\text{marg}} = \mathbb{E}_{x_t \sim \beta, a_t \sim \mu}[\rho_t \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q^\pi (x_t,a_t)] \qquad \qquad \qquad (4)
$$
limiting distribution $ \beta(x) = lim_{t \rarr \infty} p(x_t = x|x_0, \mu)$ with behavior policy $\mu$



여기서 중요한 점  두가지.

1. $Q^u$ 대신에 $Q^\pi$. 따라서 $Q^\pi$ 를 추정해야 한다.
2. importance weight의 product 가 없고 marginal importance weight $\rho_t$ 를 추정할 필요가 있다.



#### Multi-Step Estimation of The State-Action Value Function



- 이 논문에서는 Retrace(Munos et al., 2016) 방법을 사용해서 $Q^\pi {x_t, a_t}$ 를 추정한다.


$$
Q^{\text{ret}} (x_t,a_t) = r-t + \gamma \bar{\rho}_{t+1} [ Q^{\text{ret}}(x_{t+1},a_{t+1})-Q(x_{t+1},a_{t+1})] + \gamma V(x_{t+1}) \qquad \qquad  \qquad (5)
$$


$\bar{\rho}_t$ 는 truncated importance weight 라 한다. $\bar{\rho}_t = min \left\{c,\rho_t \right\}$

$Q$ 는 $Q^\pi$ 의 current value estimate 고 $V(x) = \mathbb{E}_{a \sim \pi} Q(x,a)$ 



- **Retrace** is an **off-policy, return-based algorithm** which has **low variance** and is proven to **converge (in the tabular case) to the value function of the target policy for any behavior policy**, see Munos et al. (2016)

- $Q$ 를 계산 하기 위해  discrete action space인 경우 **"two heades"를 갖는 convolutional neural network** 적용했다. ( $Q_{\theta_v} (x_t, a_t)$ 와 $\pi_\theta (a_t|x_t)$ 를 동시에 추정하기 위해 ) 
- **As Retrace uses multistep returns**, it can significantly **reduce bias** in the estimation of the policy gradient



- critic $Q_{\theta_v} (x_t, a_t)$ 를 학습하기 위해 $Q^{\text{ret}}(x_t,a_t)$ 를 target으로 mean squard error loss를 사용했고 parameter $\theta_v$ 를 업데이트 하기 위해 다음가 같은 standard gradient를 사용했다.
  $$
  \left( Q^{\text{ret}}(x_t,a_t) - Q_{\theta_v} (x_t, a_t)\right ) \bigtriangledown_{\theta_v} Q_{\theta_v} (x_t,a_t) \qquad \qquad \qquad (6)
  $$
  

**The purpose of the multi-step estimator $Q^{\text{ret}}$** 

- <u>to reduce bias in the policy gradient</u>.
- <u>to enable faster learning of the critic , hence further reducing bias.</u>



#### Importance Weight Truncation with Bais Correction



- (식 4)에서 marginal importance weight는  커질 수 있어서, instability를 야기한다.
- hight variance에 대해 safe-guard를 하기 importance weight를 truncate하고 다음과 같이 $g^{\text{marg}}$ 를 분해함으로써 correction term을 도입한다.

$$
g^{\text{marg}} = \mathbb{E}_{x_t , a_t}[\rho_t \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q^\pi (x_t,a_t)] \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad\qquad \qquad \qquad \qquad \qquad\\
= \mathbb{E}_{x_t} \left [ \mathbb{E}_{a_t} [ \bar{\rho}_t \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q^\pi (x_t,a_t)] + \mathbb{E}_{a \sim \pi} \left( \left[ {\rho_t(a)-c} \over {\rho_t(a)}\right]_+ \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q^\pi (x_t,a)\right) \right] \qquad (7)
$$

- (수식 7)의 앞의 부분은 the importance weight 를 clipping 하여 gradient estimate 의 variance가 bound되게 한다.
- (수식 7)의 뒷 부분 correction term은 $\rho_t (a) > c$ 일 때 active된다.



corret term 의 $Q^\pi (x_t,a)$ 은 neural network approximation $Q_{\theta_v} (x_t,a)$ 로 모델링 한다. 



**Truncation with bias correction trick**
$$
\bar{g}^{\text{marg}} = \mathbb{E}_{x_t} \left [ \mathbb{E}_{a_t} [ \bar{\rho}_t \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q^{\text{ret}} (x_t,a_t)] + \mathbb{E}_{a \sim \pi} \left( \left[ {\rho_t(a)-c} \over {\rho_t(a)}\right]_+ \bigtriangledown_\theta log \pi_\theta (x_t|x_t)Q_{\theta_v} (x_t,a)\right) \right] \quad (8)
$$




(식 8)은 Markov process의 statioary distribution에 대해 expection을 포함하고 있는데 이것은 sampling trajectories로 approximation할 수 있다. 


$$
\hat{g}^{\text{acer}} = \bar{\rho}_t \bigtriangledown_\theta log \pi_\theta (x_t|x_t)[Q^{\text{ret}} (x_t,a_t) - V_{\theta_v}(x_t)] \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\+ \mathbb{E}_{a \sim \pi} \left( \left[ {\rho_t(a)-c} \over {\rho_t(a)}\right]_+ \bigtriangledown_\theta log \pi_\theta (x_t|x_t)[Q_{\theta_v} (x_t,a) - V_{\theta_v}(x_t)] \right)  \quad (9)
$$




#### Efficient Trust Region Policy Opimization



- The policy updates of actor-critic methods do often **exhibit high variance**
- To ensure stability, we must **limit the per-step changes to the policy**.



- **TRPO**
  -  requires repeated computation of Fisher-vector products for each update. (prohibitively expensive in large domains)

  

- **average policy network**
  - a **running average of past policies** .
  - forces the updated policy **to not deviate far from this average**.



- policy network를 distribution $f$ 와 이 distribution의 statistics $\phi _\theta (x)$ 를 generate 하는 deep neural network 로 나눈다.  즉  $f$ 가 주어지면 policy는 $\phi_\theta : \pi(\cdot |x) = f( \cdot | \phi_\theta (x))$ 에 의해 characterized 된다. 
  - 예) $f$ 는 statistics로 probability vector $\phi_\theta(x)$ 를 갖는  categorical distribution으로 선택할 수 있다.
- $\theta : \theta_a \larr \alpha \theta_a + (1-\alpha) \theta$


$$
\hat{g}^{\text{acer}} = \bar{\rho}_t \bigtriangledown_{\phi_\theta (x_t)} log f (a_t|\phi_{\theta_t}(x))[Q^{\text{ret}} (x_t,a_t) - V_{\theta_v}(x_t)] \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\+ \mathbb{E}_{a \sim \pi} \left( \left[ {\rho_t(a)-c} \over {\rho_t(a)}\right]_+ \bigtriangledown_{\phi_\theta (x_t)} log f (a_t|\phi_{\theta_t}(x))[Q_{\theta_v} (x_t,a) - V_{\theta_v}(x_t)] \right)  \quad (10)
$$


- averated policy network 가 있을 때, 제안된 trust region 업데이트는 두 단계를 거친다. 

  - 선형화된 KL divergence 제약식을 갖는 optimization 문제를 푼다
    $$
    \underset{z} {\text{minimize}} \quad  {1 \over 2} || \hat{g}^{\text{acer}} -z || ^2_2 \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad\\
    \text{subject to }\quad \bigtriangledown_{\phi_\theta (x_t)}D_{\text{KL}}[f( \cdot|\theta_a (x_t))||f(\cdot|\phi_\theta (x_t))]^Tz \le \delta \qquad (11)
    $$
    

  - 제약식이 선현이기 때문에, overall optimization problem 은 simple quadratic programming problem으로 reduce 할 있는데, 이것의 solution은 KKT codition을 사용한 closed 형태로 쉽게 derived할 수 있다.
    $$
    z^* = \hat{g}_t^\text{acer} - \text{max} \left \{ 0 , {{k^T \hat{g}_t^\text{acer} - \delta} \over {||k||^2_2}} \right\}k \qquad \qquad (12)
    $$
    







### ACER Pseudo-Code for Discrete Actions



![2](2.png)





### RESULTS ON ATARI



![1](1.png)

