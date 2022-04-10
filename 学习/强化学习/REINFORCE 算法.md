## REINFORCE 算法
#策略梯度 是一种基于梯度的强化学习方法．在 #策略梯度 的公式
$${  \begin{align} 
\frac{\partial \mathcal{J}(\theta)}{\partial \theta} =\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T-1}\left(\frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) \gamma^{t} G\left(\tau_{t: T}\right)\right)\right],  
\end{align} }$$
中, 期望可以通过采样的方法来近似。对当前策略 ${\pi_{\theta}}$, 可以随机游走采集多个轨迹 ${\tau^{(1)}, \tau^{(2)}, \cdots, \tau^{(N)}}$,每一条轨迹 ${\tau^{(n)}=s_{0}^{(n)}, a_{0}^{(n)}, s_{1}^{(n)}, a_{1}^{(n)}, \cdots}$, , 其梯度定义为 $${ \frac{\partial \mathcal{J}(\theta)}{\partial \theta} \approx \frac{1}{N} \sum_{n=1}^{N}\left(\sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t}^{(n)} | s_{t}^{(n)}\right) \gamma^{t} G_{\tau_{t: T}^{(n)}}\right) }$$结合 #随机梯度上升 算法, 我们可以每次采集一条轨迹, 计算每个时刻的梯度并更新参数, 称为 #REINFORCE 算法 [Williams, 1992]。
$$\begin{array}{ll} 
\hline 
& \text{REINFORCE算法}\\
\hline 
& 输入: 状态空间 \mathcal{S}, 动作空间 \mathcal{A}, 可微分的策略函数 \pi_{\theta}(a | s),\\
& \quad \quad \quad  折扣率 \gamma, 学习率 \alpha; \\
1& \quad 随机初始化参数 \theta; \\
2& \quad repeat \\
3& \quad \quad 根据策略 \pi_{\theta}(a | s) 生成一条轨迹 \\
 & \quad \quad \tau=s_{0}, a_{0}, s_{1}, a_{1}, \cdots, s_{T-1}, a_{T-1}, s_{T} ; \\
4& \quad \quad for\ t=0\ to\ T\ do \\
5& \quad \quad \quad 计算\ G\left(\tau_{t: T}\right); // 更新策略函数参数\\
6& \quad \quad \quad \theta \leftarrow \theta+\alpha \gamma^{t} G\left(\tau_{t: T}\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right); \\
7& \quad \quad end \\
8& \quad until\  \theta 收敛; \\
& 输出: 策略 \pi_{\theta}\\
\hline 
\end{array}
$$
## 带基准线的 REINFORCE 算法
REINFORCE算法的一个主要缺点是不同路径之间的方差很大, 导致训练不稳定，这是在高维空间中使用蒙特卡罗方法的的通病。一种减少方差的通用方法是引入一个控制变量。假设要估计函数 ${f}$ 的期望, 为了减少 ${f}$ 的方差, 我们引入一个已知期望的函数 ${g}$, 令 $${ \hat{f}=f-\alpha(g-\mathbb{E}[g]) . }$$ 因为 ${\mathbb{E}[\hat{f}]=\mathbb{E}[f]}$, 我们可以用 ${\hat{f}}$ 的期望来估计函数 ${f}$ 的期望, 同时利用函数 ${g}$ 来减小 ${\hat{f}}$ 的方差。 
函数 ${\hat{f}}$ 的方差为 $${ \operatorname{var}(\hat{f})=\operatorname{var}(f)-2 \alpha \operatorname{cov}(f, g)+\alpha^{2} \operatorname{var}(g), }$$ 其中 ${\operatorname{var}(\cdot), \operatorname{cov}(\cdot, \cdot)}$ 分别表示方差和协方差。 
如果要使得 ${\operatorname{var}(\hat{f})}$ 最小, 令 ${\frac{\partial \operatorname{var}(\hat{f})}{\partial \alpha}=0}$, 得到 ${ \alpha=\frac{\operatorname{cov}(f, g)}{\operatorname{var}(g)} . }$ 因此, $${ \begin{align} \operatorname{var}(\hat{f}) 
&=\left(1-\frac{\operatorname{cov}(f, g)^{2}}{\operatorname{var}(g) \operatorname{var}(f)}\right) \operatorname{var}(f) \\ 
&=\left(1-\operatorname{corr}(f, g)^{2}\right) \operatorname{var}(f), \end{align}}$$ 其中 ${\operatorname{corr}(f, g)}$ 为函数 ${f}$ 和 ${g}$ 的相关性。如果相关性越高, 则 ${\hat{f}}$ 的方差越小。 

#带基准线的REINFORCE算法 
在每个时刻 ${t}$, 其策略梯度为 $${ \frac{\partial \mathcal{J}_{t}(\theta)}{\partial \theta}=\mathbb{E}_{s_{t}}\left[\mathbb{E}_{a_{t}}\left[\gamma^{t} G\left(\tau_{t: T}\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right]\right] . }$$ 为了减小策略梯度的方差, 我们引入一个和 ${a_{t}}$ 无关的基准函数 ${b\left(s_{t}\right)}$, $${ \frac{\partial \hat{\mathcal{J}}_{t}(\theta)}{\partial \theta}=\mathbb{E}_{s_{t}}\left[\mathbb{E}_{a_{t}}\left[\gamma^{t}\left(G\left(\tau_{t: T}\right)-b\left(s_{t}\right)\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right]\right] . }$$ 因为 ${b\left(s_{t}\right)}$ 和 ${a_{t}}$ 无关, 有 $${ \begin{align} &\mathbb{E}_{a_{t}}\left[b\left(s_{t}\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right] \\
=&\int_{a_{t}}\left(b\left(s_{t}\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right) \pi_{\theta}\left(a_{t} | s_{t}\right) d a_{t} \\
=&\int_{a_{t}} b\left(s_{t}\right) \frac{\partial}{\partial \theta} \pi_{\theta}\left(a_{t} | s_{t}\right) d a_{t} \\
=&\frac{\partial}{\partial \theta} b\left(s_{t}\right) \int_{a_{t}} \pi_{\theta}\left(a_{t} | s_{t}\right) d a_{t} \\
=&\frac{\partial}{\partial \theta}\left(b\left(s_{t}\right) \cdot 1\right)\\
=&0 . 
\end{align}}$$注: $\int_{a_{t}} \pi_{\theta}\left(a_{t} \mid s_{t}\right) d a_{t}=1$

因此, ${\frac{\partial \hat{\mathcal{J}}_{t}(\theta)}{\partial \theta}=\frac{\partial \mathcal{J}_{t}(\theta)}{\partial \theta}}$ 。 
为了可以有效地减小方差, ${b\left(s_{t}\right)}$ 和 ${G\left(\tau_{t: T}\right)}$ 越相关越好, 一个很自然的选择是令 ${b\left(s_{t}\right)}$ 为值函数 ${V^{\pi_{\theta}}\left(s_{t}\right)}$ 。但是由于值函数末知, 我们可以用一个可学习的函数 ${V_{\phi}\left(s_{t}\right)}$ 来近似值函数, 目标函数为 $${ \mathcal{L}\left(\phi | s_{t}, \pi_{\theta}\right)=\left(V^{\pi_{\theta}}\left(s_{t}\right)-V_{\phi}\left(s_{t}\right)\right)^{2}, }$$ 其中 ${V^{\pi_{\theta}}\left(s_{t}\right)=\mathbb{E}\left[G\left(\tau_{t: T}\right)\right]}$ 也用蒙特卡罗方法进行估计。
采用随机梯度下降法, 目标函数参数 ${\phi}$ 的梯度为 $${ \frac{\partial \mathcal{L}\left(\phi | s_{t}, \pi_{\theta}\right)}{\partial \phi}=-\left(G\left(\tau_{t: T}\right)-V_{\phi}\left(s_{t}\right)\right) \frac{\partial V_{\phi}\left(s_{t}\right)}{\partial \phi} }$$ 策略函数参数 ${\theta}$ 的梯度为 $${ \frac{\partial \hat{\mathcal{J}}_{t}(\theta)}{\partial \theta}=\mathbb{E}_{s_{t}}\left[\mathbb{E}_{a_{t}}\left[\gamma^{t}\left(G\left(\tau_{t: T}\right)-V_{\phi}\left(s_{t}\right)\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right]\right] . }$$ 
算法给出了带基准线的 REINFORCE算法的训练过程。
$$\begin{array}{ll} 
\hline 
& \text{带基准线的REINFORCE算法}\\
\hline 
& 输入: 状态空间 \mathcal{S}, 动作空间 \mathcal{A}, 可微分的策略函数 \pi_{\theta}(a | s), \\
& \quad \quad \quad 可微分 的状态值函数 V_{\phi}(s), 折扣率 \gamma, 学习率 \alpha, \beta; \\
1& \quad 随机初始化参数 \theta, \phi;\\
1& \quad repeat\\
1& \quad \quad 根据策略 \pi_{\theta}(a | s) 生成一条轨迹\\
 & \quad \quad \quad \tau=s_{0}, a_{0}, s_{1}, a_{1}, \cdots, s_{T-1}, a_{T-1}, s_{T} ;\\
1& \quad \quad for\ t=0\ to\ T\ do\\
1& \quad \quad \quad 计算 G\left(\tau_{t: T}\right);\\
1& \quad \quad \quad \delta \leftarrow G\left(\tau_{t: T}\right)-V_{\phi}\left(s_{t}\right) ;// 更新值函数参数 \\
1& \quad \quad \quad \phi \leftarrow \phi+\beta \delta \frac{\partial}{\partial \phi} V_{\phi}\left(s_{t}\right); // 更新策略函数参数 \\
1& \quad \quad \quad \theta \leftarrow \theta+\alpha \gamma^{t} \delta \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\\
1& \quad \quad end;\\
1& \quad until\ \theta 收敛;\\
& 输出: 策略 \pi_{\theta}\\
\hline 
\end{array}
$$
