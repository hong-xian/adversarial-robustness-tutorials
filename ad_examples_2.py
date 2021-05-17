# chapter 2 linear model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(root="./MNIST", train=True,
                             download=False,
                             transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root="./MNIST", train=False,
                            download=False,
                            transform=transforms.ToTensor())
train_idx = mnist_train.targets <= 1
mnist_train.data = mnist_train.data[train_idx]
mnist_train.targets = mnist_train.targets[train_idx]
test_idx = mnist_test.targets <= 1
mnist_test.data = mnist_test.data[test_idx]
mnist_test.targets = mnist_test.targets[test_idx]
train_loader = DataLoader(dataset=mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=mnist_test, batch_size=100, shuffle=True)


def epoch(loader, model, opt=None):
    total_loss, total_err = 0, 0
    for x, y in loader:
        y_prec = model(x.reshape(x.shape[0], -1))[:, 0]
        loss = nn.BCEWithLogitsLoss()(y_prec, y.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += ((y_prec > 0) * (y == 0) + (y_prec < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# linear_model = nn.Linear(784, 1)
# optim = torch.optim.SGD(linear_model.parameters(), lr=1.)
# print("train error", "train loss", "test error", "test loss", sep="\t")
# for i in range(10):
#     train_err, train_loss = epoch(train_loader, linear_model, optim)
#     test_err, test_loss = epoch(test_loader, linear_model, optim)
#     print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")

# observe one example that classifier make mistakes on
# x_test = (mnist_test.data.float()/255).reshape((len(mnist_test.data), -1))
# y_test = mnist_test.targets
# y_p = linear_model(x_test)[:, 0]
# index = (y_p > 0) * (y_test == 0) + (y_p < 0) * (y_test == 1)
# num = index.sum().item()
# if num > 0:
#     print("the number of prediction mistake", num)
#     plt.imshow(x_test[index][0].reshape(28, 28).numpy(), cmap="gray")
#     plt.title("True label is :{}".format(y_test[index][0]))
#     print("True label is :{}".format(y_test[index][0]))
#     print("predicted label is :{}".format(1 if y_p[index][0] > 0 else 0))
#
# # generate ad example
# epsilon = 0.2
# delta = epsilon * linear_model.weight.detach().sign().reshape(28, 28)
# plt.figure()
# plt.imshow(delta, cmap="gray")


def epoch_ad(loader, model):
    total_loss, total_err = 0, 0
    for x, y in loader:
        x = x - (2 * y.float()[:, None, None, None] - 1) * delta
        y_prec = model(x.reshape(x.shape[0], -1))[:, 0]
        loss = nn.BCEWithLogitsLoss()(y_prec, y.float())
        total_err += ((y_prec > 0) * (y == 0) + (y_prec < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# print("test_err with ad", "test_loss with ad", sep="\t")
# print(epoch_ad(test_loader, linear_model))
# # (0.8321513002364066, 3.264669228952827)

# fig, ax = plt.subplots(5, 5, figsize=(6, 6))
# choice = np.random.choice(len(mnist_test.data), 25, replace=False)
# for i in range(25):
#     ax[i % 5][i//5].imshow(1 - (x_test[choice[i]].reshape(28, 28) - (2 * y_test[choice[i]] - 1) * delta), cmap="gray")
#     y_p = linear_model((x_test[choice[i]].reshape(28, 28) -
#                         (2 * y_test[choice[i]] - 1) * delta).reshape(-1, 784))[:, 0].item()
#     y_p = 1 if y_p > 0 else 0
#     ax[i % 5][i // 5].set_title("true:{}".format(y_test[choice[i]]) +
#                                 " prec:{}".format(y_p), fontsize=8, y=1.001)
#     ax[i % 5][i//5].axis("off")
# plt.show()


# training robust linear model
def epoch_robust(loader, model, epsilon, opt=None):
    total_loss, total_err = 0., 0.
    for x, y in loader:
        y_prec = model(x.reshape(x.shape[0], -1))[:, 0] - (2*y.float()-1) * epsilon * model.weight.norm(1)
        loss = nn.BCEWithLogitsLoss()(y_prec, y.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += ((y_prec > 0) * (y == 0) + (y_prec < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


model = nn.Linear(784, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
epsilon = 0.2
print("rob train err", "rob train loss", "rob test err", "rob test loss", sep="\t")
for i in range(50):
    train_err, train_loss = epoch_robust(train_loader, model, epsilon, opt)
    test_err, test_loss = epoch_robust(test_loader, model, epsilon, opt)
    print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
# 0.027714	0.124368	0.018913	0.089727

# ad model on non-adversarial dataset
train_err, train_loss = epoch(train_loader, model)
test_err, test_loss = epoch(test_loader, model)
print("train err", "train loss", "test err", "test loss", sep="\t")
print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
# 0.008370	0.024126	0.004728	0.013620
# fundamental tradeoff between clean accuracy and robust accuracy

delta = epsilon * model.weight.detach().sign().view(28, 28)
plt.imshow(1-delta.numpy(), cmap="gray")

# check the ad accuracy， observe when epsilon=0.3
delta = 0.3 * model.weight.detach().sign().reshape(28, 28)
ad_train_err, ad_train_loss = epoch_ad(train_loader, model)
ad_test_err, ad_test_loss = epoch_ad(test_loader, model)
print("ad train err", "ad train loss", "ad test err", "ad test loss", sep="\t")
print(*("{:.6f}".format(i) for i in (ad_train_err, ad_train_loss, ad_test_err, ad_test_loss)), sep="\t")
# 0.110304	0.372969	0.094090	0.307462



