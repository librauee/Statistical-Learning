#  逻辑斯谛回归与最大熵模型
* logistic regression是统计学习中的经典分类方法。最大熵是概率模型学习的一个准则，推广到分类问题得到最大熵模型(maxium entropy model)
* 这两种模型都属于对数线性模型
##   逻辑斯谛回归模型
* 二项逻辑斯谛回归模型是一种分类模型，由条件概率分布P(Y|X)表示，形式为参数化的逻辑斯谛分布。
* 分类问题，可以表示成one-hot的形式，而one-hot可以认为是一种确定概率的表达。而最大熵模型，是一种不确定的概率表达，其中这个概率，是一个条件概率，是构建的特征函数生成的概率。
### 逻辑斯谛分布

* $X$是连续随机变量，$X$服从逻辑斯谛分布
$$
F(x)=P(X\leqslant x)=\frac{1}{1+\exp(-(x-\mu)/\gamma)}
$$

* 关于逻辑斯谛， 更常见的一种表达是Logistic function
$$
\sigma(z)=\frac{1}{1+\exp(-z)}
$$
* 这个函数把实数域映射到(0, 1)区间，这个范围正好是概率的范围， 而且可导，对于0输入， 得到的是0.5，可以用来表示等可能性。

### 二项逻辑斯谛回归模型



* 二项逻辑斯谛回归模型是如下的条件概率分布：(这里的$w$是对扩充的权值向量，包含参数$b$)
$$
\begin{aligned}
P(Y=1|x)&=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}\\
&=\frac{\exp(w\cdot x)/\exp(w\cdot x)}{(1+\exp(w\cdot x))/(\exp(w\cdot x))}\\
&=\frac{1}{e^{-(w\cdot x)}+1}\\
P(Y=0|x)&=\frac{1}{1+\exp(w\cdot x)}\\
&=1-\frac{1}{1+e^{-(w\cdot x)}}\\
&=\frac{e^{-(w\cdot x)}}{1+e^{-(w\cdot x)}}
\end{aligned}
$$
### 模型参数估计
* 应用极大似然估计法估计模型参数，从而得到回归模型，具体步骤为求对数似然函数，并对$L(w)$求极大值，得到$w$的估计值

$$
\begin{aligned}
L(w)&=\sum_{i=1}^N[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]\\
&=\sum_{i=1}^N[y_i(w\cdot x_i)-\log(1+\exp(w\cdot x_i))]\\
\end{aligned}
$$
* 上述过程将$P(Y=1|x)=\pi(x)$代入$L(w)$中,从而对$L(w)$求极大值，得到$w$的估计值，这样问题就变成了以对数似然函数为目标函数的最优化问题。通常采用的方法是梯度下降法以及拟牛顿法。

### 多项逻辑斯谛回归

* 假设离散型随机变量$Y$的取值集合是${1,2,\dots,K}$, 多项逻辑斯谛回归模型是
$$
\begin{aligned}
P(Y=k|x)&=\frac{\exp(w_k\cdot x)}{1+\sum_{k=1}^{K-1}\exp(w_k\cdot x)}, k=1,2,\dots,K-1\\
P(Y=K|x)&=\frac{1}{1+\sum_{k=1}^{K-1}\exp(w_k\cdot x)}\\
\end{aligned}
$$
* 上述两式和为1


## 最大熵模型

* 最大熵模型是由最大熵原理推导实现的，而最大熵原理是概率模型的一个准则。最大熵原理认为，学习概率模型时，在所有可能的概率模型中，熵最大的模型就是最好的模型。通常用约束条件来确定概率模型的集合。
### 联合熵

   * $H(X, Y) = H(X) + H(Y|X) = H(Y)+H(X|Y) = H(X|Y)+H(Y|X)+I(X;Y)$

   * 如果$X$和$Y$独立同分布，联合概率分布$P(X,Y)=P(X)P(Y)$ 

### 条件熵

   * 条件熵是最大熵原理提出的基础，最大的是条件熵，书中(定义6.3)

   * 条件熵衡量了条件概率分布的均匀性



  $$\begin{aligned}
   p^*&=\arg\max\limits_{p\in \mathcal C}H(p)\\
   &=\arg \max\limits_{p\in \mathcal C}(-\sum\limits_{x,y} {\tilde p(x)p(y|x)\log p(y|x) })
   \end{aligned} $$

### 互信息

  * 互信息(mutual information)，对应熵里面的交集，常用来描述差异性

  * 一般的，熵$H(Y)$与条件熵$H(Y|X)$之差称为互信息
*  **相关性主要刻画线性，互信息刻画非线性**
* 互信息和条件熵之间的关系
   $$
   I(x,y)=H(x)-H(x|y)=H(y)-H(y|x)
   $$

### 信息增益

   * 这个对应的是Chapter5的内容，决策树学习应用信息增益准则选择特征
   $$
   g(D,A)=H(D)-H(D|A)
   $$
   * 信息增益表示得知$X$的信息而使类$Y$的信息的不确定性减少的程度。

   * 在决策树学习中，信息增益等价于训练数据集中类与特征的互信息。

### 相对熵 (KL 散度) 

   * 相对熵(Relative Entropy)描述差异性，从分布的角度描述差异性，可用于度量两个概率分布之间的差异

  * KL散度不是一个度量，度量要满足交换性

  * KL散度满足非负性

### 最大熵模型的学习

* 最大熵模型的学习过程就是求解最大熵模型的过程。最大熵模型的学习可以形式化为约束最优的问题。自然而然想到了拉格朗日，这里用到了拉格朗日的对偶性，将原始问题转化为对偶问题，通过解对偶问题而得到原始问题的解。
* 简单来说，约束最优化问题包含$\leqslant0$，和$=0$两种约束条件
$$
\begin{aligned} 
	 \min_{x \in R^n}\quad &f(x) \\
	 s.t.\quad&c_i(x) \leqslant 0 , i=1,2,\ldots,k\\
	 &h_j(x) = 0 , j=1,2,\ldots,l
	\end{aligned}
$$
* 引入广义拉格朗日函数

$$
L(x,\alpha,\beta) = f(x) + \sum_{i=0}^k \alpha_ic_i(x) + \sum_{j=1}^l \beta_jh_j(x)
$$

* 在KKT的条件下，原始问题和对偶问题的最优值相等
$$
∇_xL(x^∗,α^∗,β^∗)=0\\
∇_αL(x^∗,α^∗,β^∗)=0\\
∇_βL(x^∗,α^∗,β^∗)=0\\
α_i^∗c_i(x^*)=0,i=1,2,…,k\\
c_i(x^*)≤0,i=1,2,…,k\\
α^∗_i≥0,i=1,2,…,k\\
h_j(x^∗)=0,j=1,2,…,l
$$
* 前面三个条件是由解析函数的知识，对于各个变量的偏导数为0，后面四个条件就是原始问题的约束条件以及拉格朗日乘子需要满足的约束,第四个条件是**KKT的对偶互补条件**


* 回到最大熵模型的学习，书中详细介绍了约束最优化问题
* 在$L(P, w)$对$P$求偏导并令其为零解得
$$
P(y|x)=\exp{\left(\sum_{i=1}^{n}w_if_i(x,y)+w_0-1\right)}=\frac{\exp{\left(\sum\limits_{i=1}^{n}w_if_i(x,y)\right)}}{\exp{\left(1-w_0\right)}}
$$
* 因为$\sum\limits_{y}P(y|x)=1$，然后得到模型

$$
P_w(y|x)=\frac{1}{Z_w(x)}\exp{\sum\limits_{i=1}^{n}w_if_i(x,y)}\\
$$
$$
其中，Z_w(x)=\sum_{y}\exp({\sum_{i=1}^{n}w_if_i(x,y))}
$$
* 这里$Z_w(x)$先用来代替$\exp(1-w_0)$,$Z_w$是归一化因子

* 并不是因为概率为1推导出了$Z_w$的表达式，这样一个表达式是凑出来的，意思就是遍历$y$的所有取值，求分子表达式的占比

* 对偶函数的极大化等价于最大熵模型的极大似然估计
* 已知训练数据的经验分布$\widetilde {P}(X,Y)$,条件概率分布$P(Y|X)$的对数似然函数表示为

$$L_{\widetilde {P}}(P_w)=\log\prod_{x,y}P(y|x)^{\widetilde {P}(x,y)}=\sum \limits_{x,y}\widetilde {P}(x,y)\log{P}(y|x)
$$


* 当条件分布概率$P(y|x)$是最大熵模型时

$$
\begin{aligned}
L_{\widetilde {P}}(P_w)&=\sum \limits_{x,y}\widetilde {P}(x,y)\log{P}(y|x)\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x,y)\log{(Z_w(x))}\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x)P(y|x)\log{(Z_w(x))}\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}\sum_{y}P(y|x)\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}
\end{aligned}
$$

* 推导过程用到了$\sum\limits_yP(y|x)=1$
