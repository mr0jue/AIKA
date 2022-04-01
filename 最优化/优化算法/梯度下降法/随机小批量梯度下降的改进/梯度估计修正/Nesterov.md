Nesterov 加速梯度（Nesterov Accelerated Gradient， NAG）是一种对动量法的改进 [Nesterov, 2013; Sutskever et al., 2013]，也称为Nesterov动量法（Nesterov Momentum）．
扩展了动量方法. 顺着惯性方向计算未来可能位置处的梯度而非当前位置的梯度, 提前量的设计让算法有了对其那方环境的预判能力.