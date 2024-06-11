import torch


def get_timepoints(
    t_span_min: int, t_span_max: int, time_scale: int = 30
) -> torch.Tensor:
    """
    Expand the time span into a sequence of time points

    Args:
        t_span_min (int): minimum time span
        t_span_max (int): maximum time span
        time_scale (int): time scale factor

    Returns:
        torch.Tensor: sequence of time points
    """
    return torch.linspace(
        t_span_min, t_span_max, int(time_scale * (t_span_max - t_span_min))
    )
