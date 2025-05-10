# The-optimization-of-reconstructed-dose-distribution-map-using-the-Encoder-and-Decoder-network
利用基于Encoder-Decoder网络架构对时间反转法重建的剂量分布图进行优化
1. dataset.zip为所用的数据集，该数据集包含两个子数据集，一个数据集为时间反转法（TR）重建的声压分布图，一个数据集为真实数据集Ground Truth。其中。每个数据集中包含三个.mat文件，每个.mat文件包含四十张图片，每张图片表示对应的声压分布图
2. Inverse_Imaging_Problem_Solution.m表示Encoder-Decoder网络优化时间反转法（TR）重建的声压分布图，数据集为120张，其中，119张作为训练集（包含验证集），最后一张用于网络模型测试，网络架构的设计类似于U-net结构
3. cal_error.m文件表示误差计算function,用于计算测试集的误差，计算指标包括：平均相对误差、最大相对误差等重要网络模型评价指标
