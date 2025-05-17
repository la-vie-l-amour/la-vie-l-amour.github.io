参考另一个 https://sites.google.com/view/deep-rl-bootcamp/lectures
# CS285
## Lecture 10 (use model to control , planning)
### Optimal Control and Planning

✈*三个术语区分 Optimal control , trajectory optimization , planning：*
trajectory optimization 可以看作 Optimal control 中的一种，Optimal control 可以看成选择controls ，优化 reward 或者最小化 cost。trajectory optimization 优化输出涉及到一系列状态和动作，某种程度上planning 和 trajectory optimization一样。
轨迹采样
事实上所有的强化学习问题都可以解决 Optimal control 的问题。

✈*closed-loop 和 open-loop*
![[Pasted image 20230817145727.png]]

根据模型是deterministic or stochastic ，open-loop又可以划分。
如果从基于模型的角度来看，open-loop planning应该是更加合理的方法，因为既然知道了模型的dynamic的情况，那么就不需要环境的反馈，agent直接进行推断就可以了。但是在实际基于模型的学习中，模型通常都是不完美的，如果仅仅在开始做规划，那么则可能造成非常大的累积误差，而closed-loop learning则可以根据每一步的反馈修正模型与规划的结果，从而使算法效果更好。
✈*Algorithms for open-loop planning*

Stochastic optimization ( called : black box optimization）
对优化问题进行抽象，原问题为
$$ a_1,\cdots, a_T = \arg \max_{a_1,\cdots, a_T} J(a_1,\cdots, a_T)$$
抽象为：$$A = \arg\max_{A}J(A)$$
为得到最优的$A$，最简单的算法是==Random shooting method==,也即是通过guess & check选出比较优的$A$
> 1. Pick $A_1,\cdots,A_N$ from some distribution (e.g., uniform)
> 2. choose $A_i$ based on $\arg\max_iJ(A_i)$

Random shooting method 是得不到最优解的，而且以来于采样的运气，所以对这个方法进行改进，对distribution进行更新，也即==Cross-entropy method(CEM)==,这两个算法都是black box optimization algorithms,其流程为：

> 1. sample  $A_1,\cdots,A_N$ from $p(A)$，$p(A)$ is pdf of distribution,typically use Gaussian distribution
> 2. evaluate $J(A_1),\cdots,J(A_N)$
> 3. pick the elites $A_{i_1},\cdots,A_{i_{M}}$ with the highest value, where $M < N$
> 4. refit $p(A)$ to the elites $A_{i_1},\cdots,A_{i_{M}}$

- what's the upside?
1. very fast if parallelized
2. Extremely simple

- what's the problem?
1. very harsh dimensionality limit
2. only open-loop planning

❓extension to CEM ：==CAM-ES== ,它是sort of like CEM with momentum ,CEM和CAM-ES都是一种evolution strategy。 

❓进化算法/演化算法(evolution algorithm)
请看这一章，有参考链接[[Evolution Algorithm]]

✈*Algorithms for closed-loop planning*
 Monte Carlo tree search(MCTS)，它主要是离散的状态，当然也可以用于连续状态。

### Trajectory Optimization with Derivatives
也即通过微分的方式进行轨迹优化，这里的微分是指对动态模型和监督信号进行求导。

|   | control | RL |
|:----:|:----:|:----:|
| 状态state | $x_t$ | $s_t$ |
| 动作action | $u_t$ | $a_t$ |
| 监督信号 | $c(x_t,u_t)$ | $r(s_t,a_t)$     |
| 动态模型 | $x_{t+1}\sim f(x_t,u_t)$ | $s' \sim p(s'|s,a)$    |

隐变量是一种未直接观测到但被用来解释数据关系的变量。
#### LQR (Linear quadratic regulator)
用于transition model is deterministic，也即动态模型是确定的，即$x_t = f(x_{t-1},u_{t-1})$ , 对问题进行formulate，采用控制理论中的符号:
$$\min_{u_1,\cdots, u_T}\sum_{t=1}^Tc(x_t,u_t) \quad  \color{#0F0} s.t. \color{#FFF}\  x_t = f(x_{t-1},u_{t-1})$$
将条件替换目标函数中的$x_t$，将问题转换为无约束问题，目标函数变为:
$$\min_{u_1,\cdots, u_T}c(x_1,u_1)+c(f(x_1,u_1),u_2)+\cdots+c(f(f(\cdots)\cdots),u_T)  $$
其中动态模型是有关状态和动作的线性$\color{#F00}Linear$表示:
$$f(x_t,u_t) = F_t\begin{bmatrix} x_t \\ u_t\end{bmatrix} + f_t$$而监督信号表示为状态和动作的$\color{#F00}quadratic$近似:
$$c(x_t,u_t) = \frac{1}{2}\begin{bmatrix}x_t \\ u_t\end{bmatrix}^T C_t\begin{bmatrix}x_t \\ u_t\end{bmatrix} + \begin{bmatrix}x_t \\ u_t\end{bmatrix}^T c_t$$
这便是为何称此算法为$\color{#F00} Linear\ quadratic\ regulator$.
参数$C_t = \begin{bmatrix}C_{x_t,x_t} & C_{x_t,u_t} \\ C_{u_t,x_t} & C_{u_t,u_t} \end{bmatrix}$ , $c_{t} = \begin{bmatrix}c_{x_t} \\ c_{u_t}\end{bmatrix}$ ,$F_t\ ,f_t$ 均已知.
>必须要说明的是通常我们认为$C_t$是对称矩阵symmetric matrix ,这一点对之后的公式推导很重要.

因为$C_t$ 是对称矩阵，所以$C_t^T = C_t$ ,也即 $$\begin{bmatrix}C_{x_t,x_t}^T & C_{u_t,x_t} ^T\\ C_{x_t,u_t}^T & C_{u_t,u_t}^T \end{bmatrix} = \begin{bmatrix}C_{x_t,x_t} & C_{x_t,u_t} \\ C_{u_t,x_t} & C_{u_t,u_t} \end{bmatrix}$$
所以显而易见 $$C_{x_t,u_t}^T = C_{u_t,x_t} \tag{1} $$ $$C_{u_t,x_t}^T = C_{x_t,u_t } \tag{2}$$  $$C_{x_t,x_t}^T =  C_{x_t,x_t} \tag{3}$$$$C_{u_t,u_t}^T =  C_{u_t,u_t} \tag{4}$$

---
**求解**： 目标是对上述无约束优化问题进行求解，求得最优动作序列，已知初始状态，目标状态goal state，动态模型，cost function（其实就是监督信号）。

那就求导白，无约束优化问题，也无需拉格朗日乘子转换，直接求导，但变量$u_1,\cdots, u_T$ 一共T个，该对谁先求导呢？可以明显看出，$u_T$ 只影响目标函数的最后一项$c(f(f(\cdots)\cdots),u_T)$，$u_{T-1}$ 影响目标函数的后两项，依次类推，所以我们从 $u_T$ 开始进行求导，因为只有最后一项和$u_T$有关，所以目标函数可写为:
$$Q(x_T,u_T) = const + \underbrace{\frac{1}{2}\begin{bmatrix}x_T\\ u_T\end{bmatrix}^T C_T\begin{bmatrix}x_T \\ u_T\end{bmatrix} + \begin{bmatrix}x_T \\ u_T\end{bmatrix}^T c_T}_{c(x_T,u_T)} \tag{5}$$
这里将目标函数写为$Q(x_T,u_T)$，就是按照RL中对于状态价值函数$Q(s,a)$的定义写的，只不过这里的不是$r(s,a)$，而是cost function $c(s,a)$，其实是一样的。
将$c(x_T,u_T)$展开，其中$\color{#FF0}u_T^TC_{u_T,x_T}x_T$和$\color{#FF0}x_T^TC_{x_T,u_T}u_T$ 互为转置，可通过式子(1)(2)进行论证，同时，$c(x_T,u_T)$是个scalar，所以其实$\color{#FF0}u_T^TC_{u_T,x_T}x_T=x_T^TC_{x_T,u_T}u_T$ ，一个数的转置等于它自己。
$$\begin{align}
c(x_T,u_T)&= \frac{1}{2}(x_T^TC_{x_T,x_T}x_T+\color{#FF0}u_T^TC_{u_T,x_T}x_T + x_T^TC_{x_T,u_T}u_T \color{#FFF} +u_T^TC_{u_T,u_T}u_T) +x_T^Tc_{x_T}+u_T^Tc_{u_T}\\
& = \frac{1}{2}(x_T^TC_{x_T,x_T}x_T+\color{#FF0}2x_T^TC_{x_T,u_T}u_T \color{#FFF} +u_T^TC_{u_T,u_T}u_T) +x_T^Tc_{x_T}+u_T^Tc_{u_T}
\end{align}$$
对$u_T$求导，令其为0，这里涉及到矩阵求导，参看笔记矩阵微分，采用的分母布局（Denominator layout），是Scalar By Vector的形式，推导时用到了式(1),(4)
$$\begin{align}
\nabla_{u_T}Q(x_T, u _T)  
&= (x_T^TC_{x_T,u_T})^T +\frac{1}{2} (C_{u_T,u_T}+C_{u_T,u_T}^T)u_T+c_{u_T}  \\ 
&= C_{u_T,x_T}x_T + C_{u_T,u_T}u_T+ c_{u_T}  = 0 \tag{6}
\end{align}$$
可以解出
$$
u_T  = -C^{-1}_{u_T,u_T}(C_{u_T,x_T}x_T+c_{u_T}) 
$$
令 $K_T = -C^{-1}_{u_T,u_T}C_{u_T,x_T} ,\quad k_T = -C^{-1}_{u_T,u_T}c_{u_T}$ ，它们均为常数，则$u_T$ 可以表示为
$$u_T = K_Tx_T + k_T$$
上式关于$u_T$的表达式，是使得目标函数$Q(x_T,u_T)$最小的那个动作，可以替换目标函数式(5)，将其中的$u_T$替换为关于$x_T$的表达式，得出$V(x_T)$，且$V(x_T) = \arg\,\min_{u_T}Q(x_T,u_T)$
$$\begin{align}
V(x_T) =& const + \frac{1}{2}\begin{bmatrix}x_T\\ K_Tx_T + k_T\end{bmatrix}^T C_T\begin{bmatrix}x_T \\ K_Tx_T + k_T\end{bmatrix} + \begin{bmatrix}x_T \\ K_Tx_T + k_T\end{bmatrix}^T c_T \\
=&const + \frac{1}{2}x_T^TC_{x_T,x_T}x_T+ \frac{1}{2} x_T^TC_{x_T,u_T}K_Tx_T +  \color{#0FF}\frac{1}{2} x_T^TC_{x_T,u_T}k_T \color{#FFF}+ \frac{1}{2} x_T^TK_T^TC_{u_T,x_T}x_T+\color{#0FF}\frac{1}{2} k_T^TC_{u_T,x_T}x_T \color{#FFF}+ \\
&\frac{1}{2}x_T^TK_T^TC_{u_T,u_T}K_Tx_T+\color{#F00}\frac{1}{2}x_T^TK_T^TC_{u_T,u_T}k_T+ \frac{1}{2}k_T^TC_{u_T,u_T}K_Tx_T\color{#FFF}+x_T^Tc_{x_T} + x_T^TK_T^Tc_{u_T} \\
=&const + \frac{1}{2}x^T_T(C_{x_T,x_T}+C_{x_T,u_T}K_T+K_T^TC_{u_T,x_T}+K_T^TC_{u_T,u_T}K_T)x_T + x_T^T(C_{x_T,u_T}k_T+K_T^TC_{u_T,u_T}k_T+c_{x_T}+K_T^Tc_{u_T})
 \end{align}
$$
红色的两部分是相等的，绿色的两部分也是相等的，具体原因和上面一样，因为互为转置，且是个scalar。还有两部分是相等的（第一个绿色前面和后面的那一项），不过这部分没有用到。
令
$$V_T = C_{x_T,x_T}+C_{x_T,u_T}K_T+K_T^TC_{u_T,x_T}+K_T^TC_{u_T,u_T}K_T$$
$$v_T = C_{x_T,u_T}k_T+K_T^TC_{u_T,u_T}k_T+c_{x_T}+K_T^Tc_{u_T}$$
需要说明的是$V_T$是个symmatric matrix，（和$K_T$是否symmatric无关）所以$V_T^T = V_T$，这个后续有用到。
则$V(x_T)$ 可以表示为
 $$V(x_T) = const + \frac{1}{2}x^T_TV_Tx_T + x_T^Tv_T $$
T时刻在采取了最优动作$u_T$损失变为了上式。



T-1时刻，$u_{T-1}$影响的是后两步，在T-1时刻的目标函数
$$Q(x_{T-1},u_{T-1}) = const + \underbrace{\frac{1}{2}\begin{bmatrix}x_{T-1}\\ u_{T-1}\end{bmatrix}^T C_{T-1}\begin{bmatrix}x_{T-1} \\ u_{T-1}\end{bmatrix} + \begin{bmatrix}x_{T-1} \\ u_{T-1}\end{bmatrix}^T c_{T-1}}_{c(x_{T-1},u_{T-1})} + V(f(x_{T-1},u_{T-1}))$$
毫无疑问，$V(f(x_{T-1},u_{T-1}))$是$c(x_T,u_T)$，当然也不全是，const那部分不全是，使用$u_{T-1}$和$x_{T-1}$表示$V(x_T)$ ,$$x_{T} = f(x_{T-1},u_{T-1}) = F_{T-1}\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix} + f_{T-1}$$
$$V(x_T) = V(f(x_{T-1},u_{T-1})) = \frac{1}{2}(\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix}^TF_{T-1}^TV_TF_{T-1}\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix}+\color{#F00}\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix}^TF_{T-1}^TV_Tf_{T-1}+f_{T-1}^TV_TF_{T-1}\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix}\color{#FFF}+f_{T-1}^TV_Tf_{T-1})+\begin{bmatrix} x_{T-1} \\ u_{T-1}\end{bmatrix}^TF_{T-1}^Tv_T+f_{T-1}^Tv_T$$
红色的两个部分是相同的，前面已经提到$V_T$是symmatric matrix，所以显而易见，这两者是相同的。
目标函数变为$$Q(x_{T-1},u_{T-1}) =const +\frac{1}{2}\begin{bmatrix}x_{T-1}\\ u_{T-1}\end{bmatrix}^T Q_{T-1}\begin{bmatrix}x_{T-1} \\ u_{T-1}\end{bmatrix} + \begin{bmatrix}x_{T-1} \\ u_{T-1}\end{bmatrix}^T q_{T-1} $$
其中 $Q_{T-1} = F_{T-1}^TV_TF_{T-1}+C_{T-1}$ , $q_{T-1} = F_{T-1}^TV_Tf_{T-1}+F_{T-1}^Tv_T+c_{T-1}$
求导, 这里对比式(5),(6),可以直接得出导数
$$\nabla_{u_{T-1}}Q(x_{T-1},u_{T-1}) = Q_{u_{T-1},x_{T-1}}x_{T-1}+Q_{u_{T-1},u_{T-1}}u_{T-1}+q_{u_{T-1}}=0$$
可以解出
$$
u_{T-1} = -Q^{-1}_{u_{T-1},u_{T-1}}(Q_{u_{T-1},x_{T-1}}x_{T-1}+q_{u_{T-1}}) 
$$
令 $K_{T-1} = -Q^{-1}_{u_{T-1},u_{T-1}}Q_{u_{T-1},x_{T-1}} ,\quad k_{T-1} = -Q^{-1}_{u_{T-1},u_{T-1}}q_{u_{T-1}}$ ，它们均为常数，则$u_{T-1}$ 可以表示为
$$u_{T-1} = K_{T-1}x_{T-1} + k_{T-1}$$
依次类推，Backward recursion 流程可以描述为
>for t = T to 1:
>>**1** 从当前时刻到最终时刻的损失（状态$x_t$采取动作$u_t$）可以写为 $$Q(x_{t},u_{t}) =const +\frac{1}{2}\begin{bmatrix}x_{t}\\ u_{t}\end{bmatrix}^T Q_{t}\begin{bmatrix}x_{t} \\ u_{t}\end{bmatrix} + \begin{bmatrix}x_{t} \\ u_{t}\end{bmatrix}^T q_{t}$$
>>其中 $Q_{t} = F_{t}^TV_{t+1}F_{t}+C_{t}$ , $q_{t} = F_{t}^TV_{t+1}f_{t}+F_{t}^Tv_{t+1}+c_{t}$
>>**2** 求使损失最小的$u_t$ $$u_t\leftarrow \arg\min_{u_t}Q(x_t,u_t) = K_tx_t+k_t$$
>>其中$K_{t} = -Q^{-1}_{u_{t},u_{t}}Q_{u_{t},x_{t}} ,\quad k_{t} = -Q^{-1}_{u_{t},u_{t}}q_{u_{t}}$
>>**3** 将最优动作$u_t$带入损失，得到在状态$x_t$从当前时刻到最终时刻的总损失$$V(x_t) = const + \frac{1}{2}x^T_tV_tx_t + x_t^Tv_t $$
>>其中$V_t = Q_{x_t,x_t}+Q_{x_t,u_t}K_t+K_t^TQ_{u_t,x_t}+K_t^TQ_{u_t,u_t}K_t$,$v_t = Q_{x_t,u_t}k_t+K_t^TQ_{u_t,u_t}k_t+q_{x_t}+K_t^Tq_{u_t}$

Forward recursion
>For t = 1 to T:
>> **1** 计算动作： $u_t = K_tx_t+k_t$
>> **2** 计算状态：$x_{t+1} = f(x_t,u_t)$


#### LQG（Linear quadratic gaussian regulator)
用于transition model is stochastic,也可以说是stochastic dynamics 
LQG只能用于closed-loop ,而LQR既可以是open-loop，也可以是closed-loop
动态模型那里，应用了KF(Kalman filter) ,所以这里需要填补，KF的知识，同时KF 又有扩展卡尔曼滤波(extend Kalman filter) 和 无迹卡尔曼滤波 (unscented Kalman filter)，
黎卡曼方程 Riccati方程
Hamilton-Jacobi-Bellman equation



#### DDP/iLQR
主要用于非线性例子Nonlinear case, DDP全称为Differential Dynamic Programming， iLQR全称为iterative linear quadratic regulator,如果是gaussian setting ,有时也被称为iLQG.
iLQR是监督信号是用泰勒二阶近似，动态模型是用泰勒一阶近似
DDP是监督信号用泰勒二阶近似，动态模型则是用泰勒二阶近似
[Deriving Differential Dynamic Porgramming (imgeorgiev.com)](http://www.imgeorgiev.com/2023-02-01-ddp/#eq:bellman-discrete)

|   | RL | LQR | LQG |  iLQR/DDP |
|:----:|:----:|:----:|:----:|:----:|
| 监督信号 | Network, ect | quadratic | quadratic | Non-Linear |
| 动态模型 | Network, ect | Linear | Linear Gaussian | Non-Linear |





## Lecture 11 (learn a  model)
why does naive approach not work?
the effect of distributional shift in model-based RL

### uncertainty in model-based RL
### model-based RL with complex observations


## Lecture 12 (policy learning with mbrl)
## Lecture 13

# paper

## （Guided Policy Search, GPS）
## （Policy Gradient Search, PGS)

## （Kalman filters）

# Chapter 8 
有关suttton的这本书，参见up主shuhuai008的讲解
## MCTS

具体的流程操作，参看
[【双语字幕】MCTS蒙特卡洛树搜索算法详细步骤解释_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1fY4y1876L/?spm_id_from=333.337.search-card.all.click&vd_source=2c0021dfb98aee58f7a63ef2d9ad3b48)
有关(Upper Confidence Bounds,UCB)[PR Reasoning Ⅱ：Bandit问题与 UCB / UCT / AlphaGo - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/218398647)

MCTS 流程
>1. 



## 优先遍历（Prioritized sweeping）





# Ref
\[1] CS285 lecture10-14 , slides2022  [Index of /deeprlcourse/static/slides (berkeley.edu)](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/)
[CS 285 (berkeley.edu)](https://rail.eecs.berkeley.edu/deeprlcourse-fa21/) 2021
\[2] paper called model-based rl a survey 
\[3] chapter 8  of *Reinforcement learning, An Introduction Second Edition*
\[4] course CS294-112
\[5][LQR,iLQR,DDP控制论经典算法（MBRL基础知识）](https://blog.csdn.net/weixin_40056577/article/details/104270668)
[初探强化学习 (boyuai.com)](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)

