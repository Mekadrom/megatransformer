from torch.optim.lr_scheduler import LambdaLR

def get_noam_scheduler(optimizer, warmup_steps, d_model, last_epoch=-1):
    """
    Noam learning rate scheduler from "Attention is All You Need"
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def lr_lambda(current_step: int):
        current_step = max(1, current_step)
        
        rate = min(
            current_step ** (-0.5),
            current_step * warmup_steps ** (-1.5)
        )
        return d_model ** (-0.5) * rate
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
