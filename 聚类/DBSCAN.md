DBSCAN 全称'Density-Based Spatial Clustering of Application with Noise'，DBSCAN 是一种著名的密度聚类算法，它基于一组“邻域”（neighborhood）参数（ϵ，MinPts）来刻画样本分布的紧密程度．相比于K-Means，DBSCAN 不需要知道簇的数量，但需要确定距离 r 和 MinPts 
# 定义
对于给定数据集$D={x_1,x_2,...,x_m}$：
- ϵ-邻域：对$x_j\in D$，其ϵ-邻域包含样本集D中与$x_j$的距离不大于ϵ的样本，即$N_ϵ(x_j)={x_i\in D|dist(x_i，x_j)\leq ϵ}$;
- 核心对象（core object）：若的ϵ-邻域至少包含MinPts个样本，即$|N_ϵ(x_j)|\geq MinPts$，则$x_j$是一个核心对象；
- 密度直达（directly density-reachable）：若$x_j$位于$x_i$的ϵ-邻域中，且$x_i$是核心对象，则称$x_j$由$x_i$密度直达；
*密度直达关系通常不满足对称性．*
- 密度可达（density-reachable）：对$x_j$与$x_i$，若存在样本序列$p_1,p_2,...,p_n$，其中$p_1=x_i, p_n=x_j$且$p(i+1)$由$p(i)$密度直达，则称$x_j$由$x_i$密度可达；
*密度可达关系满足直递性，但不满足对称性．*
- 密度相连（density-connected）：对$x_i$与$x_j$，若存在$x_k$使得$x_i$与$x_j$均由$x_k$密度可达，则称$x_i$与$x_j$密度相连．
*密度相连关系满足对称性*
- 簇：由密度可达关系导出的最大的密度相连样本集合．形式化地说，给定邻域参数（ϵ，MinPts），簇$C\subseteq D$是满足连接性和最大性的非空样本子集．
-- 连接性（connectivity）：$x_i\in C,x_j\in C$ =>$x_i$与$x_j$密度相连
-- 最大性（maximality）：$x_i\in C$, $x_j$由$x_i$a密度可达=>$x_j\in C$

D中不属于任何簇的样本被认为是噪声（noise）或异常（anomaly）样;
# 伪代码
1． 首先确定半径r和minPoints
2． 从一个没有被访问过的任意数据点开始，以这个点为中心 r为半径的圆内包含的点的数量 是否 >=minPoints：
    是：该点被标记为central point
    否：该点被标记为noise point
3． 一个noise point 是否存在于某个central point为半径的圆内：
	是：该点被标记为边缘点
	否：该点仍为noise point
4． 重复步骤2、3直到所有的点都被访问过
