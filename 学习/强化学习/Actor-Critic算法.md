在REINFORCE算法中，每次需要根据一个策略采集一条完整的轨迹，并计算这条轨迹上的回报。这种采样方式的方差比较大，学习效率也比较低。我们可以借鉴时序差分学习的思想，使用动态规划方法来提高采样的效率，即从状态开始 ${s}$ 的总回报可以通过当前动作的即时奖励 ${r\left(s, a, s{\prime}\right)}$ 和下一个状态 ${s{\prime}}$ 的值函数来近似估计。

#演员-评论员算法 ( #Actor-Critic Algorithm) 是一种结合 #策略梯度 和 #时序差分学习 的强化学习方法。其中演员 (actor) 是指 #策略函数 ${\pi_{\theta}(s, a)}$, 即学习一个策略来得到尽量高的回报, 评论员 (critic) 是指 #值函数 ${V_{\phi}(s)}$, 对当前策略的值函数进行估计, 即评估 actor 的好坏。借助于值函数, Actor-Critic 算法可以进行单步更新参数, 不需要等到回合结束才进行更新。 

在 Actor-Critic 算法中的策略函数 ${\pi_{\theta}(s, a)}$ 和值函数 ${V_{\phi}(s)}$ 都是待学习的函数, 需要在训练过程中同时学习。

假设从时刻 ${t}$ 开始的回报 ${G\left(\tau_{t: T}\right)}$, 我们用下面公式近似计算。 $${ \hat{G}\left(\tau_{t: T}\right)=r_{t+1}+\gamma V_{\phi}\left(s_{t+1}\right), }$$ 其中 ${s_{t+1}}$ 是 ${t+1}$ 时刻的状态, ${r_{t+1}}$ 是即时奖励。

在每步更新中, 分别进行策略函数 ${\pi_{\theta}(s, a)}$ 和值函数 ${V_{\phi}(s)}$ 的学习。一方面, 更新参数 ${\phi}$ 使得值函数 ${V_{\phi}\left(s_{t}\right)}$ 接近于估计的 #真实回报 ${\hat{G}\left(\tau_{t: T}\right)}$, $${ \min _{\phi}\left(\hat{G}\left(\tau_{t: T}\right)-V_{\phi}\left(s_{t}\right)\right)^{2} }$$ 另一方面, 将值函数 ${V_{\phi}\left(s_{t}\right)}$ 作为基函数来更新参数 ${\theta}$, 减少策略梯度的方差。 $${ \theta \leftarrow \theta+\alpha \gamma^{t}\left(\hat{G}\left(\tau_{t: T}\right)-V_{\phi}\left(s_{t}\right)\right) \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) }$$
在每步更新中, 演员根据当前的环境状态 ${s}$ 和策略 ${\pi_{\theta}(a | s)}$ 去执行动作 ${a}$, 环境状态变为 ${s{\prime}}$, 并得到即时奖励 ${r}$ 。评论员 (值函数 ${V_{\phi}(s)}$ ) 根据环境给出的真实奖励和之前标准下的打分 ${\left(r+\gamma V_{\phi}\left(s{\prime}\right)\right)}$, 来调整自己的打分标准, 使得自己的评分更接近环境的真实回报。演员则跟据评论员的打分, 调整自己的策略 ${\pi_{\theta}}$, 争取下次做得更好。开始训练时, 演员随机表演, 评论员随机打分。通过不断的学习, 评论员的评分越来越准, 演员的动作越来越好。

算法给出了actor-critic算法的训练过程。
$$\begin{array}{ll} 
\hline 
& \text{actor-critic算法}\\
\hline 
& 输入: 状态空间 \mathcal{S}, 动作空间 \mathcal{A};可微分的策略函数 \pi_{\theta}(a | s);\\
& \quad \quad \quad 可微分的状态值函数 V_{\phi}(s);折扣率 \gamma, 学习率 \alpha>0, \beta>0;\\
1& \quad 随机初始化参数 \theta, \phi ; \\
2& \quad repeat \\
3& \quad \quad 初始化起始状态 s; \\
4& \quad \quad \lambda=1; \\
5& \quad \quad repeat \\
6& \quad \quad \quad 在状态 s, 选择动作 a=\pi_{\theta}(a | s); \\
7& \quad \quad \quad 执行动作 a, 得到即时奖励 r 和新状态 s{\prime}; \\
8& \quad \quad \quad \delta \leftarrow r+\gamma V_{\phi}\left(s{\prime}\right)-V_{\phi}(s) ; \\
9& \quad \quad \quad \phi \leftarrow \phi+\beta \delta \frac{\partial}{\partial \phi} V_{\phi}(s); \\
10& \quad \quad \quad \theta \leftarrow \theta+\alpha \lambda \delta \frac{\partial}{\partial \theta} \log \pi_{\theta}(a | s); \\
11& \quad \quad \quad \lambda \leftarrow \gamma \lambda ; \\
12& \quad \quad \quad s \leftarrow s; \\
13& \quad \quad until\ s 为终止状态; \\
14& \quad until\ \theta 收敛; \\
& 输出: 策略 \pi_{\theta}\\
\hline 
\end{array}
$$


虽然 #带基准线的REINFORCE算法 也同时学习策略函数和值函数，但是它并不是一种 Actor-Critic 算法。因为其中值函数只是用作基线函数以减少方差，并不用来估计回报（即评论员的角色）。