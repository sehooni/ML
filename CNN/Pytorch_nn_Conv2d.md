# CNN pytorch 관련 내용

## Pytorch nn.Conv2d
    conv = torch.nn.Conv2d(in_channels=, out_channels=, kernel_size=,
                 stride = 1, padding=0, dilation=1, groups=1,bias=True)
이때 `stride, padding, dilation, groups` 는 default value이다.
 
ex) 입력채널 1 / 출력채널 1 / 커널크기 3*3
    conv = nn.Conv2d(1, 1, 3)

input type : torch.Tensor
input shape : (N * C * H * W)
              (batch_size, channel, height, width)

## Output Volume Caculations
    Output size = (input size - filter size + (2 * padding))/stride + 1

다음 예제들을 수기로 풀면 다음과 같다.
### 예제 1)
    input image size : 227 * 227
    filter size : 11 * 11
    stride = 4
    padding = 0
    output image size = ?
    공식에 따라 계산하면 (227-11+2*0)/4 + 1 = 55
                        55 * 55

### 예제 2)
    input image size : 64 * 64
    filter size : 7 * 7
    stride = 2
    padding = 0
    output image size = ?
    공식에 따라 계산하면 (64-7+2*0)/2 + 1 = 29.5 = 29
                        29 * 29

### 예제 3)
    input image size : 32 * 32
    filter size : 5 * 5
    stride = 1
    padding = 2
    output image size = ?
    공식에 따라 계산하면 (32-5+2*2)/1 + 1 = 32
                        32 * 32

### 예제 4)
    input image size : 32 * 64
    filter size : 5 * 5
    stride = 1
    padding = 0
    output image size = ?
    공식에 따라 계산하면 (32-5+2*0)/1 + 1 = 28, (64-5+2*0)/1 + 1 = 60
                        28 * 60

### 예제 5)
    input image size : 64 * 32
    filter size : 3 * 3
    stride = 1
    padding = 1
    output image size = ?
    공식에 따라 계산하면 (64-3+2*1)/1 + 1 = 64, (32-3+2*1)/1 + 1 = 32
                        64 * 32

위 예제들을 pytorch를 이용하여 계산할 수 있다.
<Pytorch_nn_Conv2d.py>(https://github.com/sehooni/ML-Pytorch/blob/master/CNN/Pytorch_nn_Conv2d.py)에서 확인 가능하다.
