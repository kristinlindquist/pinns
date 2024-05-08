import torch
import math, os, sys

from hnn.models import MLP, HNN
from hnn.utils import L2_loss


def train(args, data):
    diff_model = MLP(args.input_dim, args.hidden_dim, args.input_dim)
    model = HNN(
        args.input_dim, differentiable_model=diff_model, field_type=args.field_type
    )
    optim = torch.optim.Adam(
        model.parameters(), args.learn_rate, weight_decay=args.weight_decay
    )

    x = data["x"].clone().detach().requires_grad_()
    test_x = data["test_x"].clone().detach().requires_grad_()
    dxdt = data["dx"].clone().detach()
    test_dxdt = data["test_dx"].clone().detach()

    # vanilla train loop
    stats = {"train_loss": [], "test_loss": []}
    for step in range(args.total_steps + 1):

        # train
        model.train()
        optim.zero_grad()
        dxdt_hat = model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()

        # test
        model.eval()
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # log
        stats["train_loss"].append(loss.item())
        stats["test_loss"].append(test_loss.item())
        # print(
        #     "step {}, train_loss {:.4e}, test_loss {:.4e}".format(
        #         step, loss.item(), test_loss.item()
        #     )
        # )

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2

    print(
        "Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}".format(
            train_dist.mean().item(),
            train_dist.std().item() / math.sqrt(train_dist.shape[0]),
            test_dist.mean().item(),
            test_dist.std().item() / math.sqrt(test_dist.shape[0]),
        )
    )

    return model, stats
