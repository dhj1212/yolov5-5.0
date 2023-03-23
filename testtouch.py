import torch
import torchvision
if __name__ == '__main__':
    print('1')
    print(torch.__version__)  # 查看torch版本
    print('2')
    print(torchvision.__version__)  # 查看torchvision版本
    print('3')
    print(torch.cuda.is_available())  # 查看torch下cuda是否可用
    print('4')
    print(torch.cuda.device_count())  # 查看#GPU驱动数量
    print('5')
    print(torch.cuda.get_device_name())  # 查看#GPU驱动动名称
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(a)