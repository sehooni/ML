import torch
import torch.nn as nn

# 이를 통해 우리가 `Pytorch_nn_Conv2d.md`에서 수기로 진행한 예제와 동일함을 확인 할 수 있다.

# 예제 1
print('<예제 1>')
# conv layer 설정
conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
print(conv) # conv layer가 제대로 구축되었는지 확인

# input layer 설정
input = torch.Tensor(1, 1, 227, 227)
print(input.shape) # input의 shape 확인

# out 설정
out = conv(input)
print(out.shape) # out의 shape 확인

# 예제 2
print('<예제 2>')
# conv layer 설정
conv2 = nn.Conv2d(1, 1, 7, stride=2, padding=0)
print(conv2) # conv layer가 제대로 구축되었는지 확인

# input layer 설정
input2 = torch.Tensor(1, 1, 64, 64)
print(input2.shape) # input의 shape 확인

# out 설정
out2 = conv2(input2)
print(out2.shape) # out의 shape 확인

# 예제 3
print('<예제 3>')
# conv layer 설정
conv3 = nn.Conv2d(1, 1, 5, stride=1, padding=2)
print(conv3) # conv layer가 제대로 구축되었는지 확인

# input layer 설정
input3 = torch.Tensor(1, 1, 32, 32)
print(input3.shape) # input의 shape 확인

# out 설정
out3 = conv3(input3)
print(out3.shape) # out의 shape 확인

# 예제 4
print('<예제 4>')
# conv layer 설정
conv4 = nn.Conv2d(1, 1, 5, stride=1, padding=0)
print(conv4) # conv layer가 제대로 구축되었는지 확인

# input layer 설정
input4 = torch.Tensor(1, 1, 32, 64)
print(input4.shape) # input의 shape 확인

# out 설정
out4 = conv4(input4)
print(out4.shape) # out의 shape 확인

# 예제 5
print('<예제 5>')
# conv layer 설정
conv5 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
print(conv5) # conv layer가 제대로 구축되었는지 확인

# input layer 설정
input5 = torch.Tensor(1, 1, 64, 32)
print(input5.shape) # input의 shape 확인

# out 설정
out5 = conv5(input5)
print(out5.shape) # out의 shape 확인