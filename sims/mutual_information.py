from __future__ import division
import numpy as np
import pdb
import sklearn
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from cfg_mk import cfg_mk


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor.squeeze(0)
        self.y = y_tensor.squeeze(0)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def nnDecode(xtrain, ytrain, xtest, ytest, linearDecode=True):

    torch.manual_seed(0)
    use_cuda = False

    N, D_in = xtrain.shape[0], xtrain.shape[1],
    H, D_out = 32, 1  # TODO: pass these in as parameters

    x = torch.from_numpy(xtrain).float()  # TODO: check why I need .float()
    y = torch.from_numpy(ytrain).float().reshape(N, 1)

    dataset = CustomDataset(x, y)
    loader = DataLoader(dataset=dataset, batch_size=N, shuffle=True)

    H1 = 64
    H2 = 64
    H3 = 64
    dropout_linear = 0.8
    dropout = 0.5

    if linearDecode:
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.Dropout(dropout_linear),
            torch.nn.Linear(H1, H2),
            torch.nn.Dropout(dropout_linear),
            torch.nn.Linear(H2, H3),
            torch.nn.Dropout(dropout_linear),
            torch.nn.Linear(H3, D_out)
        )
    # if linearDecode:
    #     model = torch.nn.Sequential(
    #         torch.nn.Linear(D_in, D_out)
    #     )
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(H1, H2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(H2, H3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(H3, D_out)
        )

    if use_cuda:
        model.to('cuda')

    model = model.train()

    # loss_fn = torch.nn.MSELoss(reduction='sum')  # TODO: change from mse to cross entropy loss
    # loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change from mse to cross entropy loss
    loss_fn = torch.nn.BCEWithLogitsLoss()

    learning_rate = 5e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # , weight_decay=0.001)

    # overfit for now on training set
    eps = 0
    for t in range(3000):  # 3000 # 1000
        for i, (xbatch, ybatch) in enumerate(loader):
            if use_cuda:
                xbatch = xbatch.cuda()
                ybatch = ybatch.cuda()
            y_pred = model(xbatch)
            # pdb.set_trace()
            loss = loss_fn(y_pred + eps, ybatch)
            # if t % 10 == 0:

            if i % 20 == 0:
                print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # TODO: evaluate on the test set
    model = model.eval()
    xtest = torch.from_numpy(xtest).float().squeeze(0)  # TODO: check why I need .float()
    ytest = torch.from_numpy(ytest).float().reshape(ytest.shape[0], 1)  # TODO: remove numbers, make variable
    if use_cuda:
        xtest = xtest.cuda()
        ytest = ytest.cuda()

    y_pred_test = model(xtest)
    loss_test = loss_fn(y_pred_test + eps, ytest)
    accuracy = sum((y_pred_test > 0) == ytest).float() / len(ytest)
    return loss_test, accuracy
