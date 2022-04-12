import numpy as np
import tensorflow as tf

# 7. [그림 4-14]에서 나머지 8개 화소의 값을 계산하시오.

@tf.function
def forward():
  in_channels = 3 # 3 for RGB, 32, 64, 128, ...
  out_channels = 1 # 128, 256, ...

  ones_3d = np.array([
      [[1,2,0],[1,2,3],[1,2,0]],
      [[2,1,1],[1,0,0],[3,1,1]],
      [[0,0,1],[1,0,0],[0,1,0]],
  ])

  weight_4d = np.array([
      [[0,0,1],[0,2,0],[0,0,0]],
      [[0,0,0],[0,2,2],[1,0,0]],
      [[0,0,0],[1,2,0],[0,0,1]],
  ])
  strides_2d = [1, 1, 1, 1]

  in_3d = tf.constant(ones_3d, dtype=tf.float32)
  filter_4d = tf.constant(weight_4d, dtype=tf.float32)

  in_width = int(in_3d.shape[0])
  in_height = int(in_3d.shape[1])

  filter_width = int(filter_4d.shape[0])
  filter_height = int(filter_4d.shape[1])

  input_3d   = tf.reshape(in_3d, [1, in_height, in_width, in_channels])
  kernel_4d = tf.reshape(filter_4d, [filter_height, filter_width, in_channels, out_channels])

  #output stacked shape is 3D = 2D x N matrix
  output_3d = tf.nn.conv2d(input_3d, kernel_4d, strides=strides_2d, padding='SAME')
  return output_3d
result = forward()

print(result)