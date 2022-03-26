# 2022/03/25 16:07
# have a good day!
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


model = load_model('手写体训练模型/model1.h5')
k = np.load('data.npy')

r = model.predict(k)
k = k.reshape(28, 28) * 255.0
plt.imshow(k, cmap='gray')
plt.show()