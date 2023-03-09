# Found in the EEnets pytorch implementation. 
import torch.nn.functional as F

def loss_v2(num_ee, pred, target, conf, cost, lambda_coef=1.0):
    """loss version 2

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.
    """
    cum_pred = [None] * num_ee + [pred[num_ee]]
    cum_cost = [None] * num_ee + [cost[num_ee]]
    pred_loss = F.nll_loss(cum_pred[-1].log(), target)
    cum_loss = pred_loss + lambda_coef * cum_cost[-1].mean()
    for i in range(num_ee - 1, -1, -1):
        cum_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cum_pred[i + 1]
        cum_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cum_cost[i + 1]
        pred_loss = F.nll_loss(cum_pred[i].log(), target)
        cost_loss = cum_cost[i].mean()
        cum_loss += pred_loss + lambda_coef * cost_loss

    return cum_loss, 0, 0