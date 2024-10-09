
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class PeakZoomModel(nn.Module):
    def __init__(self):
        super(PeakZoomModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 1))
        self.peak_attention = nn.Parameter(torch.randn(1, 24, 12))  # Assuming peak attention size matches your data
        input_features = 1 * 24 * 12  # 1 * 24 * 12 = 288
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 1)  # 输出一个预测值，代表是否需要注意力峰值叠加处理

    def compute_batch_similarity(self, X1, X2):
        batch_similarities = []
        for i in range(X1.size(0)):  # 遍历每个 batch
            X1_flattened = X1[i].view(-1)
            X2_flattened = X2[i].view(-1)
            similarity12 = F.cosine_similarity(X1_flattened.unsqueeze(0), X2_flattened.unsqueeze(0), dim=0)
            batch_similarity = similarity12
            batch_similarities.append(batch_similarity)
        return torch.stack(batch_similarities)  # (batch_size, 1)

    def forward(self, X1, X2):
        #####输入是(B, N, F, T), 要改为(B, F, T, N)
        X1 = X1.permute(0, 2, 3, 1)
        X2 = X2.permute(0, 2, 3, 1)

        X1_pooled = self.maxpool(X1)
        X2_pooled = self.maxpool(X2)

        similarities = self.compute_batch_similarity(X1_pooled, X2_pooled)  # (B, 1)

        # 使用神经网络预测是否需要注意力峰值叠加处理
        similarity_mean = similarities.mean(dim=0, keepdim=True)
        x = similarity_mean.view(similarity_mean.size(0), -1)
        x = F.relu(self.fc1(x))
        threshold_prediction = torch.sigmoid(self.fc2(x))  # 预测是否需要峰值叠加处理，输出0-1之间的概率值

        threshold = threshold_prediction.item()  # 使用模型预测的阈值

        fused_data = torch.zeros_like(X1_pooled)

        # 在相似性计算中根据三个输入处理融合数据
        for i in range(X1.size(0)):  # 遍历每个 batch
            if similarities[i].mean() > threshold:
                attention_weights = F.softmax(self.peak_attention, dim=1)
                region_data = X1_pooled[i] + X2_pooled[i]
                region_data *= attention_weights
                fused_data[i] = region_data
            else:
                fused_data[i] = X1_pooled[i] + X2_pooled[i] 

        fused_data = fused_data.permute(0, 3, 1, 2)
        return fused_data, threshold_prediction
