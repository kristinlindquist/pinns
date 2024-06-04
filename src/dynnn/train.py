import torch
import math, os, sys
import time
import torch.nn.functional as F

from dynnn.models import DynNN
from dynnn.utils import save_model


def train(args: dict, data: dict):
    """
    Training loop
    """

    def calc_loss(dsdt, dsdt_hat, s):
        """
        Calculate the loss
        """
        loss = F.mse_loss(dsdt, dsdt_hat)

        if args.additional_loss is not None:
            new_loss = args.additional_loss(s, s + dsdt_hat * 0.01).sum()
            loss += new_loss

        return loss

    torch.set_default_device(args.device)

    model = DynNN(
        (args.n_bodies, 2, args.n_dims), args.hidden_dim, field_type=args.field_type
    )
    optim = torch.optim.Adam(
        model.parameters(), args.learn_rate, weight_decay=args.weight_decay
    )

    # batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
    s = data["x"].clone().detach().requires_grad_().to(args.device)
    test_s = data["test_x"].clone().detach().requires_grad_().to(args.device)
    dsdt = data["dx"].clone().detach().requires_grad_().to(args.device)
    test_dsdt = data["test_dx"].clone().detach().requires_grad_().to(args.device)

    # vanilla train loop
    stats = {"train_loss": [], "test_loss": []}
    for step in range(args.total_steps + 1):
        ### train ###
        model.train()
        optim.zero_grad()
        idxs = torch.randperm(s.shape[0])[: args.batch_size]
        dsdt_hat = model.forward(s[idxs])
        loss = calc_loss(dsdt[idxs], dsdt_hat, s[idxs])
        loss.backward()

        # with torch.autograd.profiler.profile() as prof:
        #     loss.backward()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        ### test ###
        model.eval()
        test_idxs = torch.randperm(test_s.shape[0])[: args.batch_size]
        test_dsdt_hat = model.forward(test_s[test_idxs])  # .detach()
        test_loss = calc_loss(test_dsdt[test_idxs], test_dsdt_hat, test_s[test_idxs])

        # log
        stats["train_loss"].append(loss.item())
        stats["test_loss"].append(test_loss.item())
        print(
            "step {}, train_loss {:.4e}, test_loss {:.4e}".format(
                step, loss.item(), test_loss.item()
            )
        )

    train_dsdt_hat = model.forward(s)
    train_dist = (dsdt - train_dsdt_hat) ** 2
    test_dsdt_hat = model.forward(test_s)
    test_dist = (test_dsdt - test_dsdt_hat) ** 2

    print(
        "Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}".format(
            train_dist.mean().item(),
            train_dist.std().item() / math.sqrt(train_dist.shape[0]),
            test_dist.mean().item(),
            test_dist.std().item() / math.sqrt(test_dist.shape[0]),
        )
    )

    save_model(model)

    return model, stats
