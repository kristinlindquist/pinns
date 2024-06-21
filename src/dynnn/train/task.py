from typing import Callable
import torch
import math, os, sys
import time
import torch.nn.functional as F

from dynnn.types import Dataset, ModelStats, SaveableModel, TrainingArgs, TransformY
from dynnn.utils import get_logger

logger = get_logger(__name__)


def train_task_model(
    task_model: SaveableModel,
    args: TrainingArgs,
    data: Dataset,
    transform_y: TransformY,
) -> ModelStats:
    """
    Training loop for the "outer task"

    Args:
        task_model: task model
        args: training arguments
        data: dataset
        transform_y: (optional) transformation function for getting y (actual) from x
    """
    optim = torch.optim.Adam(
        task_model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )

    # batch_size x (time_scale*t_span_max) x n_bodies x 2 x n_dims
    x = data.x.clone().detach().requires_grad_()
    test_x = data.test_x.clone().detach().requires_grad_()
    masses = data.masses.detach()

    y = transform_y(*[v.squeeze(-2) for v in x.split(1, dim=-2)], masses).detach()
    test_y = transform_y(*[v.squeeze(-2) for v in x.split(1, dim=-2)], masses).detach()

    stats = ModelStats()

    logger.info("Training task model")
    for step in range(args.n_epochs * args.steps_per_epoch + 1):
        ### train ###
        task_model.train()
        optim.zero_grad()

        y_hat = task_model.forward(x)

        loss = F.mse_loss(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1.0)
        optim.step()
        ### end train ###

        ### test ###
        task_model.eval()
        test_y_hat = task_model.forward(test_x).detach()
        test_loss = F.mse_loss(test_y_hat, test_y)
        ### end test ###

        stats.train_loss.append(loss)
        stats.test_loss.append(test_loss)

        # callback & log stats for every step, until the first epoch
        if step % (args.steps_per_epoch // 10) == 0 or step < args.steps_per_epoch:
            if args.plot_loss_callback is not None:
                args.plot_loss_callback(stats)

            logger.info(
                "OUTER TASK Step {}, train_loss {:.4e}, test_loss {:.4e}".format(
                    step,
                    loss.item(),
                    test_loss.item(),
                )
            )

    task_model.save()

    return stats
