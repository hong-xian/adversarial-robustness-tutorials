import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pig_img = Image.open("/Users/xiujing/Desktop/adversarial/introduction/pig.jpg")
pig_img = Image.open("/home/liushuang/PycharmProjects/lab/mydata/ad/pig.jpg")
preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
pig_tensor = preprocess(pig_img)[None, :, :, :]
# print(pig_tensor.shape) torch.Size([1, 3, 224, 224])
plt.imshow(pig_tensor[0].numpy().transpose(1, 2, 0))


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.tpye_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if __name__ == "__main__":
    model = resnet50(pretrained=True).to(device)
    model.eval()
    pred = model(norm(pig_tensor).to(device))
    with open("/home/liushuang/PycharmProjects/lab/mydata/ad/imagenet_class_index.json") as f:
        imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}
    result = pred.max(dim=1)[1].item()
    print(imagenet_classes[result])
