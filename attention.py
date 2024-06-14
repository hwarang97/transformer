from torch import nn as nn
from torch import Tensor

# 어떤 하이퍼 파라미터가 필요한가, 어떤 연산이 필요한가 어떻게 구현할것인가(API를 이용하거나 없으면 직접 구현)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = Tensor() # 아마도 사이즈에 대한 정보가 필요할것으로 보임. 있다가 찾아보기
        self.key = Tensor()
        self.value = Tensor()
        self.score = self.get_score()
        self.distribution = self.get_distribution()
        self.attention = self.get_attention()

    def get_vectors(self, emb_vector):
        self.qeury, self.key, self.value = dot_product(emb_vector, matrixes)

    def get_score(self, qeury: Tensor, key: Tensor) -> Tensor:
        return Tensor()

    def get_distribution(self, score: Tensor) -> Tensor:
        return softmax(score, 1) # softmax를 어디서 가져오고, 어떤 식으로 해야 행별로 비중을 나타낼 수 있을지를 찾아보기

    def get_attention(self, distribution: Tensor, value: Tensor) -> Tensor: # 벡터가 여러개인데 list를 써야하나?
        return dot_product(distribution, value)

    def forward(self,emb_vector):
        get_vectors(emb_vector)
        score = self.score(self.query, self.key)
        distribution = self.distribution(score)
        attention_value = self.attention(distribution, self.value)
        return attention_value

