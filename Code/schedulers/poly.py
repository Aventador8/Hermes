
def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * ( 1.0 -epoch /num_epochs )**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




