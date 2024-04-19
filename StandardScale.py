class StandardScaler():
    """
        Standard Scaler
        1. 在训练模型之前，首先对数据集进行标准化处理，使用标准化后的训练数据来训练模型，
        提高精度和收敛速度。
        2. 在做出预测之前，将新的输入数据用相同的方式标准化，进行预测 - 预测数据也需要处理。
        3. 如果预测结果是具体的值，那就需要还原，如果是分类和概率，那就不需要还原了。

        1. 模型预测的是连续的数值。如果这些数值（如房价、温度、销售额等）在训练时被标准化了，
        为了使预测结果具有实际意义和可用性，通常需要将这些预测值反向转换回它们原始的数值尺度
    """
    def __init__(self):
        # 预定义mean = 0， std = 1
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        # 调用fit方法实时更新
        # 沿着第0维也就是特征维度进行fit，去计算特征的均值和标准差
        # 这里的0指的是特征的0维
        # 如果数据是一个形状为 (n_samples, n_features) 的二维数组或矩阵，self.mean 和 self.std 将每个都是一个长度为 n_features 的数组。
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        # 将data进行标准化 - 如果data是一个pytorch张量，将self.mean（numpy数组）转换为tensor，并转到GPU上
        # 作用：确保均值和标准差是和data相同的数据类型并位于同一个设备
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        # 进行归一化操作
        return (data - mean) / std

    def inverse_transform(self, data):
        """
            将标准化后的数据转换回原始尺度，先处理均值和标准差，确保和数据data有相同的类型和位于相同的设别
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        # 检查数据的最后一个维度与均值和标准差的维度是否相符，不符则进行调整
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean