import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# 生成隨機數據
data = np.random.randn(1000, 2)

# 設置SOM參數
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# 繪製SOM結果
plt.figure(figsize=(10, 10))
for i, x in enumerate(data):
    w = som.winner(x)
    plt.text(w[0] + .5, w[1] + .5, str(i), color=plt.cm.rainbow(i / 1000), fontdict={'weight': 'bold', 'size': 9})
plt.xlim([0, som.get_weights().shape[0]])
plt.ylim([0, som.get_weights().shape[1]])
plt.title('Self-Organizing Map (SOM)')
plt.show()