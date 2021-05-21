# combinatorial optimization
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_dataset = torchvision.datasets.MNIST(root="./MNIST", train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root="./MNIST", train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


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


def form_milp(model, c, initial_bounds, bounds):
    linear_layers = [(layer, bound) for layer, bound in zip(model, bounds) if isinstance(layer, nn.Linear)]
    d = len(linear_layers) - 1

    # create cvxpy variables
    z = ([cp.Variable(tuple([layer.in_features])) for layer, _ in linear_layers] +
         [cp.Variable(tuple([linear_layers[-1][0].out_features]))])
    v = [cp.Variable(tuple([layer.out_features]), boolean="True") for layer, _ in linear_layers[:-1]]

    # extract relevant matrices
    w = [layer.weight.detach().cpu().numpy() for layer, _ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer, _ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l, _) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_, u) in linear_layers]
    l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
    u0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()

    # add relu constraints
    constraints = []
    for i in range(d):
        constraints += [z[i+1] >= w[i] @ z[i] + b[i],
                        z[i+1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i+1],
                        w[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]

    constraints += [z[d+1] == w[d] @ z[d] + b[d]]
    constraints += [z[0] >= l0, z[0] <= u0]
    return cp.Problem(cp.Minimize((c @ z[d+1])), constraints), (z, v)


epsilon = 0.1
X, Y = iter(test_loader).next()
X, Y = X.to(device), Y.to(device)
# model_cnn.load_state_dict(torch.load("model_cnn.pt"))
# bounds = bound_propagation(model_cnn, ((X - epsilon).clamp(min=0), (X + epsilon).clamp(max=1)))
# print("lower bound: ", bounds[-1][0][0].detach().cpu().numpy())
# print("upper bound: ", bounds[-1][1][0].detach().cpu().numpy())
# model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))
# bounds1 = bound_propagation(model_dnn_2, ((X - epsilon).clamp(min=0), (X + epsilon).clamp(max=1)))
# print("lower bound1: ", bounds1[-1][0][0].detach().cpu().numpy())
# print("upper bound1: ", bounds1[-1][1][0].detach().cpu().numpy())
# bounds2 = bound_propagation(model_dnn_4, ((X - epsilon).clamp(min=0), (X + epsilon).clamp(max=1)))
# print("lower bound2: ", bounds2[-1][0][0].detach().cpu().numpy())
# print("upper bound2: ", bounds2[-1][1][0].detach().cpu().numpy())

model_small = nn.Sequential(Flatten(), nn.Linear(784, 50), nn.ReLU(),
                            nn.Linear(50, 20), nn.ReLU(),
                            nn.Linear(20, 10)).to(device)
# optimizer = torch.optim.Adam(model_small.parameters(), lr=0.01)
# for _ in range(20):
#     train_err, train_loss = epoch(train_loader, model_small, optimizer)
#     test_err, test_loss = epoch(test_loader, model_small)
#     print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")

model_small.load_state_dict(torch.load("mydata/model_small.pt"))
initial_bound = ((X[0:1] - epsilon).clamp(min=0), (X[0:1] + epsilon).clamp(max=1))
bounds = bound_propagation(model_small, initial_bound)
c = np.zeros(10)
c[Y[0].item()] = 1
c[2] = -1
prob, (z, v) = form_milp(model_small, c, initial_bound, bounds)
prob.solve(solver=cp.GLPK_MI, verbose=True)

# remaining to solve the bug

