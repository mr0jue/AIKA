#Sigmoid 型函数是指一类S型曲线函数，为两端饱和函数。常用的 Sigmoid 型函数有Logistic函数和Tanh函数。
Sigmoid 函数可以看成是一个“ #挤压 ”函数，把一个实数域的输入“挤压”到(0, 1)或者(-1, 1)。当输入值在0附近时，Sigmoid型函数近似为线性函数；当输入值靠近两端时，对输入进行抑制。输入越小，越接近于0；输入越大，越接近于1。这样的特点也和生物神经元类似，对一些输入会产生兴奋（输出为1），对另一些输入产生抑制（输出为0）。和 #感知器 使用的 #阶跃激活函数 相比，Logistic 函数是连续可导的，其数学性质更好。

### Logistic 函数
#Logistic 函数定义为$$\sigma(x)=\frac{1}{1+\exp (-x)}$$
因为Logistic函数的性质，使得装备了Logistic激活函数的神经元具有以下两点性质：
1）其输出直接可以看作是概率分布，使得神经网络可以更好地和统计学习模型进行结合。
2）其可以看作是一个 #软性门 （Soft Gate），用来控制其它神经元输出信息的数量。

### Tanh 函数
#Tanh 函数是也一种Sigmoid型函数。其定义为$${ \tanh (x)=\frac{\exp (x)-\exp (-x)}{\exp (x)+\exp (-x)} }$$ Tanh 函数可以看作是放大并平移的 Logistic 函数, 其值域是 ${(-1,1)}$ 。 $${ \tanh (x)=2 \sigma(2 x)-1 . }$$

Tanh函数的输出是 #零中心化的 （Zero-Centered），而Logistic函数的输出恒大于0。非零中心化的输出会使得其后一层的神经元的输入发生 #偏置偏移 （Bias Shift），并进一步使得梯度下降的收敛速度变慢。
图给出了Logistic函数和Tanh函数的形状。
![[Logistic函数和Tanh函数.png|400]]


### Hard-Logistic 函数 和 Hard-Tanh 函数
Logistic 函数和 Tanh 函数都是 Sigmoid 型函数, 具有饱和性, 但是计算开销较大。因为这两个函数都是在中间 (0附近) 近似线性, 两端饱和。因此, 这 两个函数可以通过分段函数来近似。

以 Logistic 函数 ${\sigma(x)}$ 为例, 其导数为 ${\sigma{\prime}(x)=\sigma(x)(1-\sigma(x))}$ 。Logistic 函数 在 0 附近的一阶泰勒展开 (Taylor expansion) 为 $${ g_{l}(x) \approx \sigma(0)+x \times \sigma{\prime}(0) \\ =0.25 x+0.5 }$$ 用分段来近似Logistic 函数, 得到  #Hard-Logistic 函数$$
\begin{aligned}
\operatorname{hard}-\operatorname{logistic}(x) &= \begin{cases}1 & g_{l}(x) \geq 1 \\
g_{l} & 0<g_{l}(x)<1 \\
0 & g_{l}(x) \leq 0\end{cases} \\
&=\max \left(\min \left(g_{l}(x), 1\right), 0\right) \\
&=\max (\min (0.25 x+0.5,1), 0).
\end{aligned}
$$同样, Tanh函数在 0 附近的一阶泰勒展开为 $${ g_{t}(x) \approx \tanh (0)+x \times \tanh {\prime}(0) \\ =x }$$这样 Tanh 函数也可以用分段函数 hard-tanh ${(x)}$ ( #Hard-Tanh 函数)来近似。 $${ \operatorname{hard}-\tanh (x) =\max \left(\min \left(g_{t}(x), 1\right),-1\right) \\ =\max (\min (x, 1),-1) }$$
图给出了 hard-Logistic 和 hard-Tanh 函数两种函数的形状。
![[Hard Sigmoid型激活函数.png]]