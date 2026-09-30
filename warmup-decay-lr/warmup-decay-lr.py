def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here
    if current_step < warmup_steps:
        fr = current_step / warmup_steps
        lr = base_lr * fr
        return lr

    else:
        dfr = (total_steps - current_step) / (total_steps - warmup_steps)
        dlr = base_lr * dfr
        return dlr