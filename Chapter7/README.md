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