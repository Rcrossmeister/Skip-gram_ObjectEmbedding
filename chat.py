import torch
import torch.nn as nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim) # 定义词向量层
        self.linear = nn.Linear(emb_dim, vocab_size) # 定义线性层

    def forward(self, x):
        emb = self.embeddings(x) # 获得词向量
        scores = self.linear(emb) # 通过线性层计算词的预测概率
        return scores

# 加载数据
inputs = ... # 输入词对
labels = ... # 目标词

# 初始化模型
model = SkipGram(vocab_size, emb_dim)

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i in range(len(inputs)):
        optimizer.zero_grad() # 梯度清零

        # 计算损失
        scores = model(inputs[i])
        loss = criterion(scores, labels[i])

        # 反向传播
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {}/{}, Loss: {:.4f}".format(epoch, num_epochs, loss.item()))

    word_vectors = model.embeddings.weight.detach().numpy()


# --------------------
import torch
import torch.nn as nn
import torch.optim as optim

# 定义词汇表大小和词嵌入维数
vocabulary_size = 5000
embedding_size = 300


# 定义模型
class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocabulary_size, bias=False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = nn.LogSoftmax(dim=1)(out)
        return log_probs


# 实例化模型
model = SkipGram(vocabulary_size, embedding_size)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    for inputs, targets in data:
        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        log_probs = model(inputs)
        loss = criterion(log_probs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
