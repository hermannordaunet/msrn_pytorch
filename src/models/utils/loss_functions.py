# Found in the EEnets pytorch implementation.
import torch.nn.functional as F

    for i in range(num_ee - 1, -1, -1):

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
    cum_pred = [None] * num_ee + [pred[-1]]
    cum_cost = [None] * num_ee + [cost[-1]]

    # TODO: Find out why they do .log here:
    # ASK: ^
    # log_cum_pred = cum_pred[-1].log()
    log_cum_pred = cum_pred[-1]

    pred_loss = F.nll_loss(log_cum_pred, target)
    cum_loss = pred_loss + lambda_coef * cum_cost[-1].mean()
    for i in range(num_ee - 1, -1, -1):
        cum_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cum_pred[i + 1]
        cum_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cum_cost[i + 1]
        pred_loss = F.nll_loss(cum_pred[i], target)
        cost_loss = cum_cost[i].mean()
        cum_loss += pred_loss + lambda_coef * cost_loss

    return cum_loss, 0, 0
