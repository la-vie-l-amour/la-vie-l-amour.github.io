# SVD
## 1. 去中心化（把坐标原点放在数据中心）
这里选择标准化 z-score,对每个特征进行标准化
一组数据n个样本，每个样本是m维（也即m个特征）， $X_{m\times n} = [x_1,  x_2, \cdots, x_m]^T$ ，其中$x_i$代表一个特征$x_i \in \mathbb{R}^n$, 
- 计算特征i的均值和标准差
$$\mu_i = \frac{1}{n} \sum_{k=1}^{n}x_{ik}$$
$$\sigma_i = \sqrt{\frac{1}{n-1}\sum_{k=1}^{n}(x_{ik} - \mu_i)^2}$$
- 去中心化后的数据
$$z_{ik} = \frac{x_{ik} - \mu_i}{\sigma_i}$$

那么便得到去中心化后的数据$Z_{m\times n} =  [z_1,  z_2, \cdots, z_m]^T$,其中$z_i \in \mathbb{R}^n$
## 2. 找坐标系（找到方差最大的方向）
- 样本的协方差矩阵
$$cov(Z) =
\begin{bmatrix}
cov(z_1, z_1) & \cdots & cov(z_m,z_1)\\
\vdots & \ddots & \vdots \\
cov(z_1,z_m) & \cdots & cov(z_m,z_m) \\
\end{bmatrix}
$$
其中$cov(z_i,z_j) = \frac{1}{n-1} \sum_{k=1}^{n} z_{ik} z_{jk} = \frac{1}{n-1}z_i^Tz_j$ ,(这里有个隐藏的点是因为前面数据已经做了去中心化，所以这里是减的是0)因此
$$cov(Z) = \frac{1}{n-1}\begin{bmatrix}
z_1^T  \\ z_2^T \\  \vdots \\ z_m^T 
\end{bmatrix}\begin{bmatrix}
z_1 & z_2 & \cdots & z_m
\end{bmatrix} = \frac{1}{n-1}ZZ^T$$
协方差矩阵是个单位矩阵，~~因为我们之前已经做了标准化~~。**这里的解释不对，这里有个前提是数据的维度是不想关的，才可以得知协方差矩阵是单位阵。但大部分情况下，这个前提是不成立的**
- 找方差最大方向，变化得到的$Y\in \mathbb{R}^{m\times n}$ 
$$Y = RSZ$$其中$R\in \mathbb{R}^{m\times m}$是正交矩阵，R是主成分,$S\in \mathbb{R}^{m\times m}$ 是对角矩阵，
那么此时数据的协方差就是$$cov(Y) = \frac{1}{n-1}(RSZ)(RSZ)^T= RS(\frac{1}{n-1}ZZ^T)S^TR^T = RSS^TR^T$$
我们将$cov(Y)$记作$C$ ,则可知$C$是个实对称矩阵，则根据**谱定理**可知:每个实对称矩阵都有$C = Q\varLambda   Q^T$的形式，$Q$包含其特征向量的正交矩阵，对角矩阵$\varLambda$包含其特征值。(实对称矩阵的特征向量是正交的)
那么可以知道$R = Q$,$\varLambda = SS^T$ ,就求出了主成分，求出了映射。
# SVD
## SVD & PCA
在求解过程中，我们需要先求出协方差矩阵，然后求协方差矩阵的特征值和特征向量才能得到主成分，如果数据维度较大，求解协方差矩阵所需的计算量会很大，那是否可以有一种方式是不用求解协方差矩阵就可以直接得到主成分的？SVD！

对上面的矩阵$Y^T$进行奇异值分解，前提是是$n>= m$$$Y^T = U_{}\Sigma V^T$$
其中 $U \in \mathbb{R}^{n\times n}, \Sigma \in \mathbb{R}^{n\times m},  V \in \mathbb{R}^{m\times m}$, $\Sigma$是对角矩阵，对角元素是奇异值，奇异值是矩阵$YY^T$的特征值开根号，$U,V$是正交矩阵
$YY^T = (U_{}\Sigma V^T)^T(U_{}\Sigma V^T) =V\Sigma^T\Sigma V^T$
$Y^TY = (U_{}\Sigma V^T)(U_{}\Sigma V^T)^T =U\Sigma\Sigma^T U^T$
因为$YY^T$和$Y^TY$都是对称矩阵，所以根据谱定理可知$V$包含$YY^T$的正交特征向量，$U$包含$Y^TY$的正交特征向量。
而上面的主成分$R$是协方差矩阵$C= \frac{1}{n-1}YY^T$的特征向量组成的矩阵，所以$R =V$

[PCA（主成分分析）原理推导 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/84946694)
[用最直观的方式告诉你：什么是主成分分析PCA_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1E5411E71z/?spm_id_from=333.788&vd_source=2c0021dfb98aee58f7a63ef2d9ad3b48)

## SVD 公式推导
https://zhuanlan.zhihu.com/p/43578482


# 特征值分解
其实就是谱定理，只有方阵且课对角化的矩阵能进行特征值分解，也即只有单纯矩阵（P12）可以进行特征值分解 
而实对称矩阵其实是单纯矩阵的一种，所以他可以进行特征值分解的
至于什么是单纯矩阵，以及其充分必要条件，看书P12
