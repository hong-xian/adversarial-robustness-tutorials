import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
num_epochs = 10
train_dataset = torchvision.datasets.MNIST(root="/home/liushuang/PycharmProjects/lab/mydata/MNIST", train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="/home/liushuang/PycharmProjects/lab/mydata/MNIST", train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 10)).to(device)
model_dnn_4 = nn.Sequential(Flatten(), nn.Linear(784, 200), nn.ReLU(),
                            nn.Linear(200, 100), nn.ReLU(),
                            nn.Linear(100, 100), nn.ReLU(),
                            nn.Linear(100, 10)).to(device)
model_cnn = nn.Sequential(nn.Conv2d(1, 32, (3, 3), padding=(1, 1)), nn.ReLU(),
                          nn.Conv2d(32, 32, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                          nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.ReLU(),
                          nn.Conv2d(64, 64, (3, 3), padding=(1, 1), stride=(2, 2)), nn.ReLU(),
                          Flatten(), nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)


def train_model(model, dataloader, cost, optim):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = cost(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Epoch: [{} / {}], loss is: {:.4f}".format(epoch+1, num_epochs, loss.item()))


def get_accuracy(model, dataloader):
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, prediction = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (prediction == labels).sum()
    accuracy = correct / total
    print("Accuracy is :{:.4f}%".format(accuracy * 100))


def fgsm(model, x, y, epsilon):
    """"construct FGSM adversarial examples """
    delta = torch.zeros_like(x, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(x + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def plot_images(x, y, yp, m):
    fig, axes = plt.subplots(m, m)
    for i in range(m):
        for j in range(m):
            axes[i][j].imshow(1-x[i*m+j][0].cpu(), cmap="gray")
            title = axes[i][j].set_title("Pred: {}".format(yp[i * m + j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i * m + j].max(dim=0)[1] == y[i * m + j] else 'r'))
            axes[i][j].set_axis_off()
        plt.tight_layout()


def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        delta = attack(model, x, y, *args)
        yp = model(x + delta)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def pgd(model, x, y, epsilon, alpha, num_iter):
    """"construct PGD adversarial examples """
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x+delta), y)
        loss.backward()
        delta.data = (delta + x.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf(model, x, y, epsilon, alpha, num_iter):
    """"construct PGD_linf adversarial examples """
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x+delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_rand(model, x, y, epsilon, alpha, num_iter, restarts):
    """"construct PGD_linf adversarial examples with random restarts"""
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(x)
    for i in range(restarts):
        delta_ini = torch.rand_like(x, requires_grad=True)
        delta = delta_ini * epsilon
        delta.retain_grad()

        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward(retain_graph=True)
            delta.data = (delta + alpha * delta.grad.data.sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()

        all_loss = nn.CrossEntropyLoss()(model(x+delta), y)
        max_delta[all_loss > max_loss] = delta[all_loss > max_loss]
        max_loss = torch.max(all_loss, max_loss)
    return max_delta.detach()


def pgd_linf_targ(model, x, y, epsilon, alpha, num_iter, y_targ):
    """construct targeted ad examples on example x """
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        yp = model(x+delta)
        loss = (yp[:, y_targ] - yp.gather(1, y[:, None])[:, 0]).sum()
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_targ2(model, x, epsilon, alpha, num_iter, y_targ):
    """construct targeted ad examples on example x """
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        yp = model(x+delta)
        loss = 2 * yp[:, y_targ].sum() - yp.sum()
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def norms(z):
    """compute norms over all but the first dimension"""
    return z.view(z.shape[0], -1).norm(dim=1)[:, None, None, None]


def pgd_l2(model, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -x), 1-x)
        # clip x+delta to [0, 1]
        delta.data *= (epsilon / norms(delta.detach()).clamp(min=epsilon))
        delta.grad.zero_()
    return delta.detach()


if __name__ == "__main__":
    print("*********", device)
    # # training model and save model
    # optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-3, weight_decay=1e-5)
    # criterion = nn.CrossEntropyLoss()
    # train_model(model_cnn, train_loader, criterion, optimizer)
    # with torch.no_grad():
    #     print("On training set: ")
    #     get_accuracy(model_cnn, train_loader)
    #     print("On test set: ")
    #     get_accuracy(model_cnn, test_loader)
    # torch.save(model_cnn.state_dict(), "model_cnn.pt")

    # # observe ad examples
    # X, Y = iter(test_loader).next()
    # X, Y = X.to(device), Y.to(device)
    # y_prec = model_dnn_2(X)
    # plot_images(X, Y, y_prec, 4)

    # # illustrate adversarial images
    # delta = fgsm(model_dnn_2, X, Y, 0.1)
    # y_prec = model_dnn_2(X + delta)
    # plot_images(X + delta1, Y, y_prec, 4)
    #
    # delta = fgsm(model_cnn, X, Y, 0.1)
    # y_prec = model_cnn(X + delta)
    # plot_images(X + delta, Y, y_prec, 4)

    # print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, fgsm, 0.1)[0])
    # print("4-layer DNN:", epoch_adversarial(model_dnn_4, test_loader, fgsm, 0.1)[0])
    # print("        CNN:", epoch_adversarial(model_cnn, test_loader, fgsm, 0.1)[0])
    # # 2-layer DNN: 0.5715
    # 4-layer DNN: 0.6893
    #         CNN: 0.526

    model_cnn.load_state_dict(torch.load("model_cnn.pt"))
    model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))
    model_cnn.load_state_dict(torch.load("model_cnn.pt"))
    X, Y = iter(test_loader).next()
    X, Y = X.to(device), Y.to(device)
    # Illustrate attacked images
    # delta_result = pgd(model_cnn, X, Y, 0.1, 1e4, 1000)
    # y_p = model_cnn(X + delta_result)
    # plot_images(X + delta_result, Y, y_p, 5)

    # # observe the magnitude of delta
    # delta = torch.zeros_like(X, requires_grad=True)
    # loss = nn.CrossEntropyLoss()(model_cnn(X + delta), Y)
    # loss.backward()
    # print(delta.grad.abs().mean().item())
    # # 3.403440859983675e-05

    # delta_linf = pgd_linf(model_cnn, X, Y, 0.1, 1e-2, 40)
    # y_plinf = model_cnn(X + delta_linf)
    # plot_images(X + delta_linf, Y, y_plinf, 5)
    # print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, pgd_linf, 0.1, 1e-2, 40)[0])
    # print("4-layer DNN:", epoch_adversarial(model_dnn_4, test_loader, pgd_linf, 0.1, 1e-2, 40)[0])
    # print("CNN:", epoch_adversarial(model_cnn, test_loader, pgd_linf, 0.1, 1e-2, 40)[0])
    # 2-layer DNN: 0.83
    # 4-layer DNN: 1.0
    # CNN: 0.9424
    # print("CNN:", epoch_adversarial(model_cnn, test_loader, pgd_linf_rand, 0.1, 1e-2, 40, 10)[0])
    # CNN: 0.9449
    # print(epoch_adversarial(model_cnn, test_loader, pgd_linf_targ, 0.2, 1e-2, 40, 0)[0])
    # # error:0.897
    # delta_result = pgd_linf_targ(model_cnn, X, Y, 0.2, 1e-2, 40, 0)
    # y_prec = model_cnn(X+delta_result)
    # plot_images(X+delta_result, Y, y_prec, 5)
    # print(epoch_adversarial(model_cnn, test_loader, pgd_linf_targ2, 0.2, 1e-2, 100, 0)[0])
    # delta_result = pgd_linf_targ2(model_cnn, X, Y, 0.2, 1e-2, 100, 0)
    # y_prec = model_cnn(X+delta_result)
    # plot_images(X+delta_result, Y, y_prec, 5)
    # delta_result = pgd_l2(model_cnn, X, Y, epsilon=2, alpha=0.1, num_iter=40)
    # yp = model_cnn(X+delta_result)
    # plot_images(X+delta_result, Y, yp, 5)
    # print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, pgd_l2, 2, 0.1, 40)[0])
    # print("4-layer DNN:", epoch_adversarial(model_dnn_4, test_loader, pgd_l2, 2, 0.1, 40)[0])
    # print("CNN:", epoch_adversarial(model_cnn, test_loader, pgd_l2, 2, 0.1, 40)[0])
    # 2-layer DNN: 0.8411
    # 4-layer DNN: 0.902
    # CNN: 0.9362




