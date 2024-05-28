import torch
import math, os, sys
import time
import torch.nn.functional as F

from dynnn.models import MLP, DynNN


def train(args: dict, data: dict):
    """
    Training loop
    """
    torch.set_default_device(args.device)

    # input_dim = n_bodies * n_dims * len([r, v])
    input_dim = args.n_bodies * args.n_dims * 2

    diff_model = MLP(input_dim, args.hidden_dim, input_dim)
    model = DynNN(
        input_dim, differentiable_model=diff_model, field_type=args.field_type
    )
    optim = torch.optim.Adam(
        model.parameters(), args.learn_rate, weight_decay=args.weight_decay
    )

    # batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
    x = data["x"].clone().detach().requires_grad_().to(args.device)
    test_x = data["test_x"].clone().detach().requires_grad_().to(args.device)
    dxdt = data["dx"].clone().detach().to(args.device)
    test_dxdt = data["test_dx"].clone().detach().to(args.device)

    # vanilla train loop
    stats = {"train_loss": [], "test_loss": []}
    for step in range(args.total_steps + 1):
        # train
        model.train()
        optim.zero_grad()
        ixs = torch.randperm(x.shape[0])[: args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        loss = F.mse_loss(dxdt[ixs], dxdt_hat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        # with torch.autograd.profiler.profile() as prof:
        #     loss.backward()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # test
        model.eval()
        test_ixs = torch.randperm(test_x.shape[0])[: args.batch_size]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs]).detach()
        test_loss = F.mse_loss(test_dxdt[test_ixs], test_dxdt_hat)

        # log
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
