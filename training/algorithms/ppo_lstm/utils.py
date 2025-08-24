def linear_lr_decay(initial_lr, current_step, total_steps):
    frac = 1.0 - float(current_step) / float(total_steps)
    return initial_lr * frac
