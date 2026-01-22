def kl_beta_schedule(epoch, max_beta, warmup_start=0, warmup_end=100):
    if epoch <= warmup_start:
        return 0.0
    if epoch < warmup_end:
        return max_beta * (epoch - warmup_start) / (warmup_end - warmup_start)
    return max_beta