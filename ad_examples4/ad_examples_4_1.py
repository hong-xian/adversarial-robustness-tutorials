import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device("cuda:1" if torch.cuda.is_available() else "gpu")
torch.manual_seed(0)

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


model_cnn = nn.Sequential(nn.Conv2d(1, 32, (3, 3), padding=(1, 1)), nn.ReLU(),
                          nn.Conv2d(32, 32, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                          nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.ReLU(),
                          nn.Conv2d(64, 64, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                          Flatten(), nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)


def fgsm(model, x, y, epsilon=0.1):
    """"construct FGSM adversarial examples """
    delta = torch.zeros_like(x, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(x + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, x, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """construct FGSM ad examples on example x"""
    if randomize:
        delta_ini = torch.rand_like(x, requires_grad=True)
        delta = delta_ini * 2 * epsilon - epsilon
        delta.retain_grad()
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x+delta), y)
        loss.backward(retain_graph=True)
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        delta = attack(model, x, y, **kwargs)
        yp = model(x + delta)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def draw_loss(model, x, y, epsilon):
    xi, yi = np.meshgrid(np.linspace(-epsilon, epsilon, 100), np.linspace(-epsilon, epsilon, 100))

    def grad_at_delta(delta):
        nn.CrossEntropyLoss()(model(x+delta), y[10:11]).backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    dir1 = grad_at_delta(torch.zeros_like(x, requires_grad=True))
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))
    # dir2是为了增加噪声？
    all_deltas = torch.tensor((np.array([xi.flatten(), yi.flatten()]).T @
                               np.array([dir2, dir1])).astype(np.float32)).to(device)
    yp = model(all_deltas.view(-1, 1, 28, 28) + x)
    zi = nn.CrossEntropyLoss(reduction="none")(yp, y[10:11].repeat(yp.shape[0])).detach().cpu().numpy()
    zi = zi.reshape(*xi.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, linewidth=0, cmap="rainbow")


if __name__ == "__main__":
    # optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-2)
    # print("train err", "test err", "adv err", sep="\t")
    # for t in range(20):
    #     train_err, train_loss = epoch(train_loader, model_cnn, optimizer)
    #     test_err, test_loss = epoch(test_loader, model_cnn)
    #     adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn, pgd_linf)
    #     if t == 10:
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = 1e-3
    #     print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
    #     0.000467	0.011500	0.968400
    # torch.save(model_cnn.state_dict(), "model_cnn.pt")

    # optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)
    # print("train err", "test err", "adv err", sep="\t")
    # for t in range(20):
    #     train_err, train_loss = epoch_adversarial(train_loader, model_cnn, pgd_linf, optimizer)
    #     test_err, test_loss = epoch(test_loader, model_cnn)
    #     adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn, pgd_linf)
    #     if t == 10:
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = 1e-4
    #     print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
    # #     0.002717	0.007000	0.021400
    # torch.save(model_cnn.state_dict(), "model_cnn_robust.pt")

    # evaluate robust models
    model_cnn.load_state_dict(torch.load("model_cnn_robust.pt"))
    # model_cnn.load_state_dict(torch.load("model_cnn.pt"))

    # print("FGSM:", epoch_adversarial(test_loader, model_cnn, fgsm)[0])
    # # FGSM: 0.0199
    # print("PGD, 40 iter:", epoch_adversarial(test_loader, model_cnn, pgd_linf, num_iter=40)[0])
    # # PGD, 40 iter: 0.0214
    # print("PGD, small_alpha: :", epoch_adversarial(test_loader, model_cnn, pgd_linf, num_iter=40, alpha=0.05)[0])
    # # PGD, small_alpha: : 0.0212
    # print("PGD, randomized: :", epoch_adversarial(test_loader, model_cnn, pgd_linf,
    #                                               num_iter=40, alpha=0.05, randomize=True)[0])
    # # PGD, randomized: : 0.0211
    X, Y = iter(test_loader).next()
    X, Y = X.to(device), Y.to(device)
    draw_loss(model_cnn, X[10:11], Y, epsilon=0.1)
    plt.title("training model")
    plt.show()
