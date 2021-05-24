import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ad_examples_4_1 import epoch, epoch_adversarial
device = torch.device("cuda:1" if torch.cuda.is_available() else "gpu")


train_dataset = datasets.MNIST(root="/home/liushuang/PycharmProjects/lab/mydata/MNIST",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root="/home/liushuang/PycharmProjects/lab/mydata/MNIST",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


model_cnn_robust_2 = nn.Sequential(nn.Conv2d(1, 32, (3, 3), padding=(1, 1)), nn.ReLU(),
                                 nn.Conv2d(32, 32, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                                 nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.ReLU(),
                                 nn.Conv2d(64, 64, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                                 Flatten(), nn.Linear(7*7*64, 100), nn.ReLU(),
                                 nn.Linear(100, 10)).to(device)


def bound_propagation(model, initial_bound):
    l, u = initial_bound
    bounds = []
    l_, u_ = 0, 0

    for layer in model:
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)

        elif isinstance(layer, nn.Linear):
            l_ = (torch.matmul(layer.weight.clamp(min=0), l.T) +
                  torch.matmul(layer.weight.clamp(max=0), u.T) + layer.bias[:, None]).T
            u_ = (torch.matmul(layer.weight.clamp(min=0), u.T) +
                  torch.matmul(layer.weight.clamp(max=0), l.T) + layer.bias[:, None]).T

        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])
            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_, u_))
        l, u = l_, u_
    return bounds


def interval_based_bound(model, c, bounds, idx):
    # require last layer to be linear
    # use idx to choose the input labeled with y
    cw = c.T @ model[-1].weight
    cb = c.T @ model[-1].bias
    l, u = bounds[-2]
    value = cw.clamp(min=0) @ l[idx].T + cw.clamp(max=0) @ u[idx].T + cb[:, None]
    return value.T


def robust_bound_error(model, x, y, epsilon):
    initial_bound = (x - epsilon, x + epsilon)
    bounds = bound_propagation(model, initial_bound)
    err = 0.
    for y0 in range(10):
        c = -torch.eye(10).to(device)
        c[y0, :] += 1
        err += ((interval_based_bound(model, c, bounds, y == y0).min(dim=1))[0] < 0).sum().item()
    return err


def epoch_robust_error(loader, model, epsilon):
    total_err = 0
    c = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        c[y0][y0, :] += 1

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        initial_bound = (x - epsilon, x + epsilon)
        bounds = bound_propagation(model, initial_bound)
        for y0 in range(10):
            lower_bound = interval_based_bound(model, c[y0], bounds, y == y0)
            total_err += (lower_bound.min(dim=1)[0] < 0).sum().item()
    return total_err / len(loader.dataset)


def epoch_robust_bound_train(loader, model, epsilon, opt=None):
    total_err = 0
    total_loss = 0

    c = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        c[y0][y0, :] += 1

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        initial_bound = (x - epsilon, x + epsilon)
        bounds = bound_propagation(model, initial_bound)
        loss = 0
        for y0 in range(10):
            if sum(y == y0) > 0:
                lower_bound = interval_based_bound(model, c[y0], bounds, y == y0)
                # use -lower_bound 让对抗样本所对应的的概率最小化
                loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y == y0]) / x.shape[0]
                total_err += (lower_bound.min(dim=1)[0] < 0).sum().item()
        total_loss += loss.item() * x.shape[0]
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# model_cnn_robust.load_state_dict(torch.load("model_cnn_robust.pt"))
# x, y = iter(test_loader).next()
# x, y = x.to(device), y.to(device)
# c = -torch.eye(10)
# c[0, :] += 1
# c = c.to(device)
# epsilon = 0.01
# initial_bound = (x - epsilon, x + epsilon)
# print(interval_based_bound(model_cnn_robust, c,
#                            bounds=bound_propagation(model_cnn_robust, initial_bound), idx=(y == 0)))
# print(epoch_robust_error(test_loader, model_cnn_robust, epsilon=0.0001))
# epsilon=0.1, 0.01, err=1; epsilon=0.001, err=0.9874, epsilon=0.0001, err=0.0157
if __name__ == "__main__":
    opt = torch.optim.Adam(model_cnn_robust_2.parameters(), lr=1e-3)
    eps_schedule = [0.0, 0.0001, 0.001, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05] + 10*[0.1]

    print("Train Eps", "Train Loss*", "Test Err", "Test Robust Err", sep="\t")
    for t in range(len(eps_schedule)):
        train_err, train_loss = epoch_robust_bound_train(train_loader, model_cnn_robust_2, eps_schedule[t], opt)
        test_err, test_loss = epoch(test_loader, model_cnn_robust_2)
        adv_err, adv_loss = epoch_robust_bound_train(test_loader, model_cnn_robust_2, 0.1)
        print(*("{:.6f}".format(i) for i in (eps_schedule[t], train_loss, test_err, adv_err)), sep="\t")
    # torch.save(model_cnn_robust_2.state_dict(), "model_cnn_robust_2.pt")
    print("PGD, 40 iter: ", epoch_adversarial(test_loader, model_cnn_robust_2, pgd_linf, num_iter=40)[0])