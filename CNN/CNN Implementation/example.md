다음 그림과 같은 모양의 학습을 진행한다.

# 그림 1. CNN Implementation
![cnn implementation](https://user-images.githubusercontent.com/84653623/163127806-018be86d-286a-4fdf-b773-4d85b1b75214.png)

이때의 코드는 다음과 같다.
    import torch
    import torch.nn as nn

    input = torch.Tensor(1, 1, 28, 28)
    conv1 = nn.Conv2d(1, 5, 5)
    pool - nn.MaxPool2d(2)

    out = conv1(input)
    out2 = pool(out)

    out.size()
    out2.size()

이와 관련된 코드는 [example.py](https://github.com/sehooni/ML-Pytorch/blob/master/CNN/CNN%20Implementation/example.py)에서 확인할 수 있다.
