# 9. 문제 8에서 보폭을 s=2로 설정했을 때 컨볼루션 결과를 쓰시오.
import numpy as np
import tensorflow as tf

def forward():
    in_channels = 1  #RGB, 32, 64, 128, ...
    out_channels = 1  # 128, 256, ...
    bias = np.array([0.5
                     ])
    ones_3d = np.array([
        [2, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 9, 9, 9, 9, 9],
        [2, 2, 2, 9, 9, 9, 9, 9],
        [2, 2, 2, 9, 9, 9, 9, 9],
        [2, 2, 2, 9, 9, 9, 9, 9]

    ])

    weight_4d = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ])
    strides_2d = [1, 2, 2, 1]   # 슬라이딩 윈도우가 한번에 이동하는 크기를 2로 설정

    in_3d = tf.constant(ones_3d, dtype=tf.float32)
    filter_4d = tf.constant(weight_4d, dtype=tf.float32)

    # 폭과 높이 지정
    in_width = int(in_3d.shape[0])
    in_height = int(in_3d.shape[1])

    filter_width = int(filter_4d.shape[0])
    filter_height = int(filter_4d.shape[1])

    # 입력 데이터를 원래 이미지의 크기로 변경
    input_3d = tf.reshape(in_3d, [1, in_height, in_width, in_channels])
    kernel_4d = tf.reshape(filter_4d, [filter_height, filter_width, in_channels, out_channels])

    # output stacked shape is 3D = 2D x N matrix
    # 여기서 zero padding 적용
    output_3d = tf.nn.conv2d(input_3d, kernel_4d, strides=strides_2d, padding='SAME')
    output_3d = tf.nn.bias_add(output_3d, bias)
    return output_3d


result = forward()
print(result)