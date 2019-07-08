# 支持向量机
## 笔记摘要
* SVM的基本模型是定义在特征空间上的间隔最大的线性分类器
* 线性可分支持向量机和线性支持向量机假设输入空间和特征空间的元素一一对应，并将输入空间中的输入映射为特征空间的特征向量；非线性支持向量机利用一个从**输入空间到特征空间的非线性映射**将输入映射为特征向量
* 支持向量机的学习策略就是**间隔最大化**，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数最小化问题
* 仿射变换是保凸变换
* 通过使用核函数可以学习非线性支持向量机，等价于隐式地在高维的特征空间中学习线性支持向量机
### 函数间隔

* 对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为
$$
\hat \gamma_i=y_i(w\cdot x_i+b)
$$
* 定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔之最小值，即
$$
\hat \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
$$
* 函数间隔可以表示分类预测的**正确性**及**确信度**

### 几何间隔

* 对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔为
$$
 \gamma_i=y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})
$$
* 定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的几何间隔之最小值，即
$$
 \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
$$
* 超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔一般是实例点到超平面的带符号的距离，当样本点被超平面正确分类时就是实例点到超平面的距离
* 如果超平面参数成比例地改变，此时超平面没有发生改变，但函数间隔按此比例改变，而几何间隔不变
### 线性可分支持向量机
* 问题描述

$$
\begin{aligned}
&\min_{w,b}\frac{1}{2}||w||^2\\
&s.t.\ \ \ y_i(w\cdot x_i+b)-1\geqslant0,i=1,2,\dots,N\\
\end{aligned}
$$

* 这是个凸二次规划问题，如果求出了上述方程的解$w^*, b^*$，就可得到分离超平面
$$
w^*\cdot x+b^*=0
$$
* 以及相应的分类决策函数
$$
f(x)=sign(w^*\cdot x+b^*)
$$



#### 对偶算法
* 通过求解对偶问题得到原始问题地最优解的优点
1. 对偶问题往往更容易求解
1. 自然引入核函数，进而推广到非线性分类问题

* 针对每个不等式约束，定义拉格朗日乘子$\alpha_i\ge0​$，定义拉格朗日函数
$$
\begin{aligned}
L(w,b,\alpha)&=\frac{1}{2}w\cdot w-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\sum_{i=1}^N\alpha_iy_i(w\cdot x_i+b)+\sum_{i=1}^N\alpha_i
\end{aligned}
$$
$$
\alpha_i \geqslant0, i=1,2,\dots,N
$$
其中$\alpha=(\alpha_1,\alpha_2,\dots,\alpha_N)^T​$为拉格朗日乘子向量
* **原始问题是极小极大问题**，根据**拉格朗日对偶性**，原始问题的**对偶问题是极大极小问题**:
$$
\max\limits_\alpha\min\limits_{w,b}L(w,b,\alpha)
$$
* 转换后的对偶问题
$$
\min\limits_\alpha \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
s.t. \ \ \ \sum_{i=1}^N\alpha_iy_i=0\\
\alpha_i\geqslant0, i=1,2,\dots,N
$$


* 根据KKT条件求解，其中$\alpha$不为零的点对应的实例为**支持向量**，通过支持向量可以求得$b$值


$$
\begin{aligned}
w^*&=\sum_{i=1}^{N}\alpha_i^*y_ix_i\\
b^*&=y_j\color{black}-\sum_{i=1}^{N}\alpha_i^*y_i(x_i\cdot x_j\color{black})
\end{aligned}
$$
* $b^*$的求解，通过$\arg\max \alpha^*$实现，因为支持向量共线，所以通过任意支持向量求解都可以



### 线性支持向量机

* 问题描述

$$
\begin{aligned}
\min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
&\xi_i\geqslant0,i=1,2,\dots,N
\end{aligned}
$$
 

* 对偶问题描述
   * 原始问题里面有两部分约束，涉及到两个拉格朗日乘子向量
$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$
通过求解对偶问题， 得到$\alpha$，然后求解$w,b$的过程和之前一样

* **线性支持向量机的解$w^*$唯一但$b^*$不一定唯一**

* 线性支持向量机是线性可分支持向量机的超集



#### 合页损失

* 最小化目标函数

$$\min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2$$

* 其中

  - 第一项是经验损失或经验风险，函数$L(y(w\cdot x+b))=[1-y(w\cdot x+b)]_+$称为合页损失，可以表示成$L = \max(1-y(w\cdot x+b), 0)$
  - 第二项是**系数为$\lambda$的$w$的$L_2$范数的平方**，是正则化项

* 书中通过定理7.4说明了用合页损失表达的最优化问题和线性支持向量机原始最优化问题的关系
$$
\begin{aligned}
\min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
&\xi_i\geqslant0,i=1,2,\dots,N
\end{aligned}
$$
 * 等价于
$$
\min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2
$$
* 证明如下
* 令合页损失$\left[1-y_i(w\cdot x+b)\right]_+=\xi_i$，合页损失非负，所以有$\xi_i\ge0$，这个对应了原始最优化问题中的**第二个约束**


* 还是根据合页损失非负，当$1-y_i(w\cdot x+b)\leq\color{red}0​$的时候，有$\left[1-y_i(w\cdot x+b)\right]_+=\color{red}\xi_i=0​$，所以有$1-y_i(w\cdot x+b)\leq\color{red}0=\xi_i$，这对应了原始最优化问题中的**第一个约束**

* 所以，在满足这**两个约束**的情况下，有
$$
\begin{aligned}
\min\limits_{w,b} &\sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2\\
\min\limits_{w,b} &\sum\limits_{i=1}^N\xi_i+\lambda\left\|w\right\|^2\\
\min\limits_{w,b} &\frac{1}{C}\left(\frac{1}{2}\left\|w\right\|^2+C\sum\limits_{i=1}^N\xi_i\right), with \  \lambda=\frac{1}{2C}\\
\end{aligned}
$$

* 合页损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190626153036271.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x5YzQ0ODEzNDE4,size_16,color_FFFFFF,t_70#pic_center)
### 非线性支持向量机

* 核技巧的想法是在学习和预测中只定义核函数$K(x,z)$，而不是显式的定义映射函数$\phi$

* 通常，直接计算$K(x,z)$比较容易， 而通过$\phi(x)$和$\phi(z)$计算$K(x,z)$并不容易。
$$
W(\alpha)=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
$$
$$
f(x)=sign\left(\sum_{i=1}^{N_s}\alpha_i^*y_i\phi(x_i)\cdot \phi(x)+b^*\right)=sign\left(\sum_{i=1}^{N_s}\alpha_i^*y_iK(x_i,x)+b^*\right) 
$$
学习是隐式地在特征空间进行的，不需要显式的定义特征空间和映射函数


#### 核函数

* 对于给定的核$K(x,z)$，特征空间$\mathcal H$和映射函数$\phi(x)$的取法并不唯一，可以取不同的特征空间，即便是同一特征空间里也可以取不同的映射

* 下面这个例子里面$\phi(x)$实现了从低维空间到高维空间的映射
$$
K(x,z)=(x\cdot z)^2\\
{X}=\R^2, x=(x^{(1)},x^{(2)})^T\\
{H}=\R^3, \phi(x)=((x^{(1)})^2, \sqrt2x^{(1)}x^{(2)}, (x^{(2)})^2)^T\\
{H}=\R^4, \phi(x)=((x^{(1)})^2, x^{(1)}x^{(2)}, x^{(1)}x^{(2)}, (x^{(2)})^2)^T\\
$$


* 核具有再生性，即满足下面条件的核称为再生核
$$
K(\cdot,x)\cdot f=f(x)\\
K(\cdot,x)\cdot K(\cdot, z)=K(x,z)
$$
* 通常所说的核函数就是**正定核函数**
* 问题描述

  * 将向量内积替换成了核函数，而SMO算法求解的问题正是该问题
  * 构建最优化问题：
$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$

  * 求解得到$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$
选择$\alpha^*$的一个正分量计算
$$
b^*=y_j-\sum_{i=1}^N\alpha_i^*y_iK(x_i,x_j)
$$
构造决策函数
$$
f(x)=sign\left(\sum_{i=1}^N\alpha_i^*y_iK(x,x_i)+b^*\right)
$$

## SMO算法

### 问题描述

$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$

这个问题中，变量是$\alpha$，一个变量$\alpha_i$对应一个样本点$(x_i,y_i)$，变量总数等于$N$



### KKT 条件

  * KKT条件是该最优化问题的充分必要条件
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

### 算法内容

整个SMO算法包括两**部分**：

1. 求解两个变量二次规划的解析方法
1. 选择变量的启发式方法

$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i, x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$



#### Part I

* 两个变量二次规划求解
* 选择两个变量$\alpha_1,\alpha_2​$，由等式约束可以得到$\alpha_1=-y_1\sum\limits_{i=2}^N\alpha_iy_i​$，所以这个问题等价于一个单变量优化问题

$$
\begin{aligned}
\min_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=&\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2\\
&-(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^Ny_i\alpha_iK_{il}+y_2\alpha_2\sum_{i=3}^Ny_i\alpha_iK_{i2}\\
s.t. \ \ \ &\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^Ny_i\alpha_i=\varsigma\\
&0\leqslant\alpha_i\leqslant C, i=1,2
\end{aligned}
$$

* 上面存在两个约束：
1. **线性**等式约束
2. 边界约束
* 根据简单的线性规划可以得出**等式约束使得$(\alpha_1,\alpha_2)$在平行于盒子$[0,C]\times [0,C]$的对角线的直线上**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627105606348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x5YzQ0ODEzNDE4,size_16,color_FFFFFF,t_70)
* 首先求沿着约束方向未经剪辑，即不考虑约束$0\leqslant\alpha_i\leqslant C$时$\alpha_2$的最优解，然后再求剪辑后的解
$$E_i=g(x_i)-y_i=(\sum_{j=1}^N\alpha_jy_jK(x_i, x_j)+b)-y_i,i=1,2$$
$E_i$为函数$g(x)$对输入的预测值与真实输出$y_i$的差
#### Part II

* 变量的选择方法

1. 第一个变量$\alpha_1$外层循环，寻找违反KKT条件**最严重**的样本点
2. 第二个变量$\alpha_2$内层循环，希望能使$\alpha_2$有足够大的变化
3. 计算阈值$b$和差值$E_i$



> 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),\dots, (x_N,y_N)}$，其中$x_i\in\mathcal X=\bf R^n, y_i\in\mathcal Y=\{-1,+1\}, i=1,2,\dots,N$,精度$\epsilon$
>
> 输出：近似解$\hat\alpha$
>
> 1. 取初值$\alpha_0=0$，令$k=0$
>
> 1. **选取**优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$，解析求解两个变量的最优化问题，求得最优解$\alpha_1^{(k+1)},\alpha_2^{(k+1)}$，更新$\alpha$为$\alpha^{k+1}$
>
> 1. 若在精度$\epsilon$范围内满足停止条件
>    $$
>    \sum_{i=1}^{N}\alpha_iy_i=0\\
>    0\leqslant\alpha_i\leqslant C,i=1,2,\dots,N\\
>    y_i\cdot g(x_i)=
>    \begin{cases}
>    \geqslant1,\{x_i|\alpha_i=0\}\\
>    =1,\{x_i|0<\alpha_i<C\}\\
>    \leqslant1,\{x_i|\alpha_i=C\}
>    \end{cases}\\
>    g(x_i)=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)+b
>    $$
>    则转4,否则，$k=k+1$转2
>
> 1. 取$\hat\alpha=\alpha^{(k+1)}$

# 习题解答

* 1.**比较感知机的对偶形式与线性可分支持向量机的对偶形式**
   * 感知机的对偶形式
   $f(x)=sign\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right),
\alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T$ 
   * 线性可分支持向量机的对偶形式
     $f(x)=sign\left(\sum_{i=1}^N\alpha_i^*y_ix_i\cdot x+b^*\right),
\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 
感知机学习算法的原始形式和对偶形式与线性可分支持向量机学习算法的原始形式和对偶形式相对应。在线性可分支持向量机的对偶形式中,$w$也是被表示为实例 $x_i$和标记$y_i$的线性组合的形式
$$w=\sum_{i=1}^{N}\alpha_iy_ix_i$$
而它们的偏置$b$的形式不同，前者$b=\sum_{i=1}^{N}\alpha_iy_i$,而后者$b^*=y_j\color{black}-\sum_{i=1}^{N}\alpha_i^*y_i(x_i\cdot x_j)$
* 2.**已知正例点$x_1=(1,2)^T$，$x_2=(2,3)^T$，$x_3=(3,3)^T$,负例点$x_4=(2,1)^T$，$x_5=(3,2)^T$，
试求最大间隔分离超平面和分类决策函数，并在图上画出分离超平面、间隔边界及支持向量**
  * 根据书中算法，计算可得$w_1=-1$,$w_2=2$,$b=-2$,即最大间隔分离超平面为
  $$-x^{(1)}+2x^{(2)}-2=0$$
  分类决策函数为
  $$f(x)=sign(-x^{(1)}+2x^{(2)}-2)$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708102330284.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x5YzQ0ODEzNDE4,size_16,color_FFFFFF,t_70#pic_center)

* 3.**线性支持向量机还可以定义为以下形式：**

$$\min_{w,b,\xi}{\frac{1}{2}\|w\|^2}+C\sum^N_{i=1}\xi_i^2\\s.t.{\quad}y_i(w{\cdot}x_i+b)\ge1-\xi_i,\,i=1,2,\cdots,N\\\xi_i\ge0,\,i=1,2,\cdots,N$$
      **试求其对偶形式**
   * 首先求得原始化最优问题的拉格朗日函数是： 
  $L(w,b,\alpha,\xi,μ)=\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i^2-\sum_{i=1}^N\alpha_i(y_i(w\cdot x_i+b-1)+\xi_i)-\sum_{i=1}^Nμ_i\xi_i$
  * 对偶问题是拉格朗日的极大极小问题。首先求$L(w,b,\alpha,\xi,μ)$对$w,b,\xi$的极小,即对该三项求偏导，得
  $$
  w=\sum_{i=1}^{N}\alpha_iy_ix_i\\
 \sum_{i=1}^N\alpha_iy_i=0\\
 2C\xi_i-\alpha_i-μ_i=0
  $$
  将上述带入拉格朗日函数，得
  $$-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-C\sum_{i=1}^N\xi_i^2+\sum_{i=1}^N\alpha_i\\
  -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\frac{1}{4C}\sum_{i=1}^N(\alpha_i+μ_i)^2+\sum_{i=1}^N\alpha_i$$
* 4.**证明内积的正整数幂函数$K(x,z)=(x{\cdot}z)^p$是正定核函数，这里$p$是正整数，$x,z{\in}R^n$**
  * 要证明正整数幂函数是正定核函数，只需证明其对应得Gram矩阵$K=[K(x_i,x_j)]_{m\times m}$是半正定矩阵
  * 对任意$c_1,c_2…c_m\in R$,有

       $$
       \begin{aligned}
       \sum_{i,j=1}^{m}c_ic_jK(x_i,x_j)\\
       =&\sum_{i,j=1}^{m}c_ic_j(x_i\cdot x_j)^p\\
       =&(\sum_{i=1}^{m}c_ix_i)(\sum_{j=1}^{m}c_jx_j)(x_ix_j)^{p-1}\\
       =&||\sum_{i=1}^{m}c_ix_i||^2(x_ix_j)^{p-1}
       \end{aligned}
       $$
       * 由于p大于等于1，该式子也大于等于0，即Gram矩阵半正定，所以正整数的幂函数是正定核函数