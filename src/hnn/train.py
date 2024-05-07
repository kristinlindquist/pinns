import torch
import math, os, sys

from hnn.models import MLP, HNN
from hnn.simulation import get_dataset
from hnn.utils import L2_loss

OUTPUT_DIM = 2


def train(args):
    nn_model = MLP(args.input_dim, args.hidden_dim, OUTPUT_DIM)
    model = HNN(
        args.input_dim, differentiable_model=nn_model, field_type=args.field_type
    )
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    data = get_dataset()

    x = torch.tensor(data["x"], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data["test_x"], requires_grad=True, dtype=torch.float32)
    dxdt = torch.tensor(data["dx"])
    test_dxdt = torch.tensor(data["test_dx"])

    print(
        "X has grad?", x.grad is not None, "Test X has grad?", test_x.grad is not None
    )

    # vanilla train loop
    stats = {"train_loss": [], "test_loss": []}
    for step in range(args.total_steps + 1):

        # train step
        dxdt_hat = model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats["train_loss"].append(loss.item())
        stats["test_loss"].append(test_loss.item())
        print(
            "step {}, train_loss {:.4e}, test_loss {:.4e}".format(
                step, loss.item(), test_loss.item()
            )
        )

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
