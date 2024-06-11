from typing import Callable
import torch
import math, os, sys
import time
import torch.nn.functional as F

from dynnn.layers.pinn import PINN
from dynnn.types import PinnStats, TrainingArgs
from dynnn.utils import save_model, save_stats


def default_loss_fn(dxdt, dxdt_hat, s, masses):
    """
    Calculate the loss
    """
    loss = F.mse_loss(dxdt, dxdt_hat)
    addtl_loss = torch.tensor(0.0)

    return loss, addtl_loss


def pinn_train(
    args: TrainingArgs,
    data: dict,
    model: torch.nn.Module,
    plot_loss_callback: Callable | None = None,
    loss_fn: Callable | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Training loop for the PINN
    """
    optim = torch.optim.Adam(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    calc_loss = loss_fn or default_loss_fn

    # batch_size x (time_scale*t_span_max) x n_bodies x 2 x n_dims
    x = data["x"].clone().detach().requires_grad_()
    test_x = data["test_x"].clone().detach().requires_grad_()
    dxdt = data["dx"].clone().detach().requires_grad_()
    test_dxdt = data["test_dx"].clone().detach().requires_grad_()
    masses = data["masses"].clone().detach().requires_grad_()

    run_id = time.time()
    counter = 0
    best_metric = float("inf")

    # vanilla train loop
    stats = PinnStats()
    for step in range(args.n_epochs * args.steps_per_epoch + 1):
        ### train ###
        model.train()
        optim.zero_grad()
        dxdt_hat = model.forward(x)
        loss, additional_loss = calc_loss(dxdt, dxdt_hat, x, masses)
        loss.backward()

        # with torch.autograd.profiler.profile() as prof:
        #     loss.backward()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        ### test ###
        model.eval()
        test_dxdt_hat = model.forward(test_x)  # .detach()
        test_loss, test_additional_loss = calc_loss(
            test_dxdt, test_dxdt_hat, test_x, masses
        )

        stats.train_loss.append(loss)
        stats.test_loss.append(test_loss)
        stats.train_additional_loss.append(additional_loss)
        stats.test_additional_loss.append(test_additional_loss)

        if step % (args.steps_per_epoch // 10) == 0 or step < args.steps_per_epoch:
            # callback & log stats for every step, until the first epoch
            if plot_loss_callback is not None:
                plot_loss_callback(stats)

            print(
                "step {}, train_loss {:.4e}, additional_loss {:.4e}, test_loss {:.4e}, test_additional_loss {:.4e}".format(
                    step,
                    loss.item(),
                    additional_loss.item(),
                    test_loss.item(),
                    test_additional_loss.item(),
                )
            )

        if step % args.steps_per_epoch == 0:
            if (step / args.steps_per_epoch) >= args.min_epochs:
                # save_stats(stats.model_dump(), run_id=run_id)

                val_metric = test_loss.item()
                if val_metric < best_metric - args.tolerance:
                    print(
                        f"Val metric improved. {val_metric} < {best_metric} - {args.tolerance}"
                    )
                    best_metric = val_metric
                    counter = 0
                    save_model(model, run_id=run_id)
                else:
                    counter += 1
                    if counter >= args.patience:
                        print("Early stopping triggered. Training stopped.")
                        break

    train_dxdt_hat = model.forward(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.forward(test_x)
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
