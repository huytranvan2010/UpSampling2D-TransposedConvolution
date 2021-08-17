import tensorflow as tf
import numpy as np

x = np.arange(12)
x = x.reshape(3, 4)

# mở rộng chiều batch size và # channels
x = x[np.newaxis, ..., np.newaxis]

# 1 pixel trong x sẽ tạo tương ứng với 4 pixels trong y
# y = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(x)
y = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)

print(y)
# Để dễ hình dung sẽ lấy ra kết quả như này cho đơn giản
print(y[0, :, :, 0])