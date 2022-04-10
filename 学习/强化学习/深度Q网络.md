为了在连续的状态和动作空间中计算值函数 ${Q^{\pi}(s, a)}$, 我们可以用一个函数 ${Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})}$ 来表示近似计算, 称为 #值函数近似 (Value Function Approximation). $${ Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) \approx Q^{\pi}(s, a), }$$ 其中 ${\boldsymbol{s}, \boldsymbol{a}}$ 分别是状态 ${s}$ 和动作 ${a}$ 的向量表示; 函数 ${Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})}$ 通常是一个参数为 ${\phi}$ 的函数, 比如神经网络, 输出为一个实数, 称为 #Q网络 （Q-network）．
如果动作为有限离散的 ${M}$ 个动作 ${a_{1}, \cdots, a_{M}}$, 我们可以让 ${\mathrm{Q}}$ 网络输出一个 ${M}$ 维向量, 其中第 ${m}$ 维表示 ${Q_{\phi}\left(s, a_{m}\right)}$, 对应值函数 ${Q^{\pi}\left(s, a_{m}\right)}$ 的近似值. $${ Q_{\phi}(\boldsymbol{s})=\left[\begin{array}{c} Q_{\phi}\left(\boldsymbol{s}, a_{1}\right) \\ \vdots \\ Q_{\phi}\left(\boldsymbol{s}, a_{M}\right) \end{array}\right] \approx\left[\begin{array}{c} Q^{\pi}\left(s, a_{1}\right) \\ \vdots \\ Q^{\pi}\left(s, a_{M}\right) \end{array}\right] . }$$我们需要学习一个参数 ${\phi}$ 来使得函数 ${Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})}$ 可以逼近值函数 ${Q^{\pi}(s, a)}$. 
如果采用蒙特卡罗方法, 就直接让 ${Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})}$ 去逼近平均的总回报 ${\hat{Q}^{\pi}(s, a)}$; 
如果采用时序差分学习方法, 就让 ${Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})}$ 去逼近 ${\mathbb{E}_{\boldsymbol{s}{\prime}, \boldsymbol{a}{\prime}}\left[r+\gamma Q_{\phi}\left(\boldsymbol{s}{\prime}, \boldsymbol{a}{\prime}\right)\right]}$. 
以 #Q学习 为例, 采用随机梯度下降, 目标函数为 $${ \mathcal{L}\left(s, a, s{\prime} | \phi\right)=\left(r+\gamma \max _{a{\prime}} Q_{\phi}\left(\boldsymbol{s}{\prime}, \boldsymbol{a}{\prime}\right)-Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})\right)^{2}, }$$ 其中 ${\boldsymbol{s}{\prime}, \boldsymbol{a}{\prime}}$ 是下一时刻的状态 ${s{\prime}}$ 和动作 ${a{\prime}}$ 的向量表示.

然而，这个目标函数存在两个问题：一是目标不稳定，参数学习的目标依赖于参数本身；二是样本之间有很强的相关性．为了解决这两个问题，[Mnih et al.,2015] 提出了一种 #深度Q网络 （Deep Q-Networks， #DQN ）．
深度Q网络采取两个措施：
一是 #目标网络冻结 （Freezing Target Networks），即在一个时间段内固定目标中的参数，来稳定学习目标；
二是 #经验回放 （Experience Replay），即构建一个 #经验池 （Replay Buffer）来去除数据相关性．经验池是由智能体最近的经历组成的数据集．
训练时，随机从经验池中抽取样本来代替当前的样本用来进行训练．这样，就打破了和相邻训练样本的相似性，避免模型陷入局部最优．经验回放在一定程度上类似于监督学习．先收集样本，然后在这些样本上进行训练．

深度Q网络的学习过程如算法所示
$$\begin{array}{ll} 
\hline 
& \text{带经验回放的深度Q网络}\\
\hline 
& 输入: 状态空间 \mathcal{S} , 动作空间 \mathcal{A} , 折扣率 \gamma , 学习率 \alpha , 参数更新间隔 C ;\\
1& \quad 初始化经验池 \mathcal{D} , 容量为 N ;\\
2& \quad 随机初始化 Q 网络的参数 \phi ;\\
3& \quad 随机初始化目标 Q 网络的参数 \hat{\phi}=\phi ;\\
4& \quad repeat \\
5& \quad \quad 初始化起始状态 s ;\\
6& \quad \quad \quad repeat \\
7& \quad \quad \quad 在状态 s , 选择动作 a=\pi^{\epsilon} ;\\
8& \quad \quad \quad 执行动作 a , 观测环境, 得到即时奖励 r 和新的状态 s{\prime} ;\\
9& \quad \quad \quad 将 s, a, r, s{\prime} 放入 \mathcal{D} 中;\\
10& \quad \quad \quad 从 \mathcal{D} 中采样 s s, a a, r r, s s{\prime} ;\\
11& \quad \quad \quad y=\left\{
	\begin{array}{cc}
	r r,& s s{\prime} 为终止状态, \\ 
	r r+\gamma \max _{a{\prime}} Q_{\phi}\left(\mathrm{ss}{\prime}, \boldsymbol{a}{\prime}\right), &否则 
	\end{array}\right. ;\\
12& \quad \quad \quad 以 \  \left(y-Q_{\phi}(\boldsymbol{s s}, \boldsymbol{a a})\right)^{2} 为损失函数来训练 Q 网络;\\
13& \quad \quad \quad s \leftarrow s{\prime} ;\\
14& \quad \quad \quad 每隔 C 步, \hat{\phi} \leftarrow \phi ;\\
15& \quad \quad until \  s 为终止状态;\\
16& \quad until \  \forall s, a, Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) 收敛;\\
& 输出: Q 网络 Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) \\
\hline 
\end{array}
$$


