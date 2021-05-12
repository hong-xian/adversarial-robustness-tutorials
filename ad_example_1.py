import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pig_img = Image.open("/Users/xiujing/Desktop/adversarial/introduction/pig.jpg")
# pig_img = Image.open("/home/liushuang/PycharmProjects/lab/mydata/ad/pig.jpg")
preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
pig_tensor = preprocess(pig_img)[None, :, :, :]
# print(pig_tensor.shape) torch.Size([1, 3, 224, 224])


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if __name__ == "__main__":
    print("************", device)
    model = resnet50(pretrained=True).to(device)
    model.eval()
    pred = model(norm(pig_tensor).to(device))
    # with open("/home/liushuang/PycharmProjects/lab/mydata/ad/imagenet_class_index.json") as f:
    #     imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}
    with open("/Users/xiujing/Desktop/adversarial/introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}
    result = pred.max(dim=1)[1].item()
    print(imagenet_classes[result])
    print(nn.CrossEntropyLoss()(pred, torch.LongTensor([341])).item())

    epsilon = 2./255
    delta = torch.zeros_like(pig_tensor, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=1e-1)
    for i in range(30):
        pred = model(norm(pig_tensor + delta))
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([341]))
        if i % 5 == 0:
            print("after {}th iteration, loss is::{}".format(i, loss.item()))

        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
    print("True class probability:", nn.Softmax(dim=1)(pred)[0, 341].item())

    max_class = pred.max(dim=1)[1].item()
    print("Predicted class:", imagenet_classes[max_class])
    print("predicted probability:", nn.Softmax(dim=1)(pred)[0, max_class].item())
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pig_tensor[0].numpy().transpose(1, 2, 0))
    axes[1].imshow((pig_tensor+delta)[0].detach().numpy().transpose(1, 2, 0))
    plt.figure()
    plt.imshow((50*delta+0.5)[0].detach().numpy().transpose(1, 2, 0))
    plt.show()


