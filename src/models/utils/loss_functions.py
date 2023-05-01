# Found in the EEnets pytorch implementation.
import torch.nn.functional as F


def loss_v1(num_ee, pred, target, conf, cost, lambda_coef=1.0):
    """loss version 1

    Arguments are
    * num_ee:   number of early exit blocks
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the fusion loss of the cross_entropy loss and cost loss.
    These loss parts are calculated in a recursive way as following:
    Prediction'_i = confidence_i * prediction_i + (1 - confidence_i) * Prediction'_(i+1)
    Cost'_i       = confidence_i * cost_i       + (1 - confidence_i) * Cost'_(i+1)
    """
    cum_pred = pred[num_ee]
    cum_cost = cost[num_ee]
    for i in range(num_ee - 1, -1, -1):
        cum_pred = conf[i] * pred[i] + (1 - conf[i]) * cum_pred
        cum_cost = conf[i] * cost[i] + (1 - conf[i]) * cum_cost

    # TODO: Find out why they do .log here:
    # ASK: ^
    # pred_loss = F.nll_loss(cum_pred.log(), target)
    pred_loss = F.nll_loss(cum_pred, target)
    cost_loss = cum_cost.mean()
    cum_loss = pred_loss + lambda_coef * cost_loss
 
    return cum_loss, pred_loss, cost_loss


def loss_v2(num_ee, pred, target, conf, cost, lambda_coef=1.0):
    """loss version 2

    Arguments are
    * num_ee:   number of early exit blocks
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.
    """
    cum_pred = [None] * num_ee + [pred[num_ee]]
    cum_cost = [None] * num_ee + [cost[num_ee]]

    # TODO: Find out why they do .log here:
    # ASK: ^
    log_cum_pred = cum_pred[-1].log()
    # log_cum_pred = cum_pred[-1]

    pred_loss = F.nll_loss(log_cum_pred, target)
    cum_loss = pred_loss + lambda_coef * cum_cost[-1].item() #.mean()
    for i in range(num_ee - 1, -1, -1):
        cum_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cum_pred[i + 1]
        cum_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cum_cost[i + 1]
        pred_loss = F.nll_loss(cum_pred[i].log(), target)
        cost_loss = cum_cost[i].mean()
        cum_loss += pred_loss + lambda_coef * cost_loss

    return cum_loss, 0, 0


def loss_v3(num_ee, pred, target, conf, cost, lambda_coef=1.0):
    """loss version 3

    Arguments are
    * num_ee:   number of early exit blocks
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.

    First loss written for DQN training loop
    """
    cum_pred = [None] * num_ee + [pred[num_ee]]
    cum_cost = [None] * num_ee + [cost[num_ee]]

    # TODO: Find out why they do .log here:
    # ASK: ^
    # log_cum_pred = cum_pred[-1]
    # log_cum_pred = cum_pred[-1]

    # pred_loss = F.nll_loss(cum_pred[-1], target)
    pred_loss = F.smooth_l1_loss(cum_pred[-1], target)
    cum_loss = pred_loss + lambda_coef * cum_cost[-1].item() #.mean()
    for i in range(num_ee - 1, -1, -1):
        cum_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cum_pred[i + 1]
        cum_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cum_cost[i + 1]
        # pred_loss = F.nll_loss(cum_pred[i].log(), target)
        pred_loss = F.smooth_l1_loss(cum_pred[i], target)
        cost_loss = cum_cost[i].mean()
        cum_loss += pred_loss + lambda_coef * cost_loss

    return cum_loss, 0, 0
