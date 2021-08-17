import tensorflow as tf
import numpy as np

x = np.random.rand(1, 3, 3, 3)

y = tf.keras.layers.Conv2DTranspose(filters=6, kernel_size=(3,3), strides=(2,2), padding='valid')(x)

print(y)
print("Output shape: ", y.shape)
print("Expected output shape: (1, 7, 7, 6)")

""" 
    (7 - kernel_size - 0)/stride + 1 = 3. 
    kernel_size, strides ảnh hưởng chính đến kích thước output
    Ở đây số filters chính là số channels đầu ra, kết hợp kích thước filers chúng ta có số parameters cần học
"""

"""
one of "valid" or "same" (case-insensitive). "valid" means no padding. 
"same" results in padding with zeros evenly to the left/right or up/down 
of the input such that output has the same height/width dimension as the input.

Cũng giống padding bên Convolutional layer, sử dụng padding='same' thì kích thước output giống với input
Bên Transposed Convolutional layer sử dụng padding='same' thì kích thước output giống với input
nhưng chut yếu mình muốn tăng kích thước nên hay để padding='valid'
"""