# Found in the EEnets pytorch implementation.
import torch.nn.functional as F


def loss_v1(pred, target, conf, cost, num_ee=0, lambda_coef=1.0):
    """loss version 1

    Arguments are
    * pred:           prediction result of each exit point.
    * target:         target prediction values.
    * conf:           confidence value of each exit point.
    * cost:           cost rate of the each exit point.
    * num_ee:         number of early exit blocks
    * lambda_coef:

    This loss function is the fusion loss of the cross_entropy loss and cost loss.
    These loss parts are calculated in a recursive way as following:
    Prediction'_i = confidence_i * prediction_i + (1 - confidence_i) * Prediction'_(i+1)
    Cost'_i       = confidence_i * cost_i       + (1 - confidence_i) * Cost'_(i+1)
    """
    cumulative_pred = pred[num_ee]
    cumulative_cost = cost[num_ee]
    for i in range(num_ee - 1, -1, -1):
        cumulative_pred = conf[i] * pred[i] + (1 - conf[i]) * cumulative_pred
        cumulative_cost = conf[i] * cost[i] + (1 - conf[i]) * cumulative_cost

    # TODO: Find out why they do .log here:
    # ASK: ^
    # pred_loss = F.nll_loss(cumulative_pred.log(), target)
    pred_loss = F.nll_loss(cumulative_pred, target)
    cost_loss = cumulative_cost.mean()
    cumulative_loss = pred_loss + lambda_coef * cost_loss

    return cumulative_loss, pred_loss, cost_loss


def loss_v2(pred, target, conf, cost, num_ee=0, lambda_coef=1.0):
    """loss version 2

    Arguments are
    * pred:           prediction result of each exit point.
    * target:         target prediction values.
    * conf:           confidence value of each exit point.
    * cost:           cost rate of the each exit point.
    * num_ee:         number of early exit blocks
    * lambda_coef:

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.
    """
    cumulative_pred = [None] * num_ee + [pred[num_ee]]
    cumulative_cost = [None] * num_ee + [cost[num_ee]]

    # TODO: Find out why they do .log here:
    # ASK: ^
    log_cumulative_pred = cumulative_pred[-1].log()
    # log_cumulative_pred = cumulative_pred[-1]

    pred_loss = F.nll_loss(log_cumulative_pred, target)
    cumulative_loss = pred_loss + lambda_coef * cumulative_cost[-1]  # .mean()
    for i in range(num_ee - 1, -1, -1):
        cumulative_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cumulative_pred[i + 1]
        cumulative_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cumulative_cost[i + 1]
        log_cumulative_pred = cumulative_pred[i].log()
        pred_loss = F.nll_loss(log_cumulative_pred, target)
        cost_loss = cumulative_cost[i].mean()
        cumulative_loss += pred_loss + lambda_coef * cost_loss

    return cumulative_loss, 0, 0


def loss_v3(pred, target, conf, cost, num_ee=0, lambda_coef=1.0):
    """loss version 3

    * pred:           prediction result of each exit point.
    * target:         target prediction values.
    * conf:           confidence value of each exit point.
    * cost:           cost rate of the each exit point.
    * num_ee:         number of early exit blocks
    * lambda_coef:

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.

    First loss written for DQN training loop
    """
    cumulative_pred = [None] * num_ee + [pred[num_ee]]
    cumulative_cost = [None] * num_ee + [cost[num_ee]]

    # TODO: Find out why they do .log here:
    # ASK: ^
    # log_cumulative_pred = cumulative_pred[-1]
    # log_cumulative_pred = cumulative_pred[-1]

    # pred_loss = F.nll_loss(cumulative_pred[-1], target)
    pred_loss = F.smooth_l1_loss(cumulative_pred[-1], target)
    cumulative_loss = pred_loss + lambda_coef * cumulative_cost[-1].item()  # .mean()
    for i in range(num_ee - 1, -1, -1):
        cumulative_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cumulative_pred[i + 1]
        cumulative_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cumulative_cost[i + 1]
        # pred_loss = F.nll_loss(cumulative_pred[i].log(), target)
        pred_loss = F.smooth_l1_loss(cumulative_pred[i], target)
        cost_loss = cumulative_cost[i].mean()
        cumulative_loss += pred_loss + lambda_coef * cost_loss

    return cumulative_loss, 0, 0


def loss_v4(pred, target, num_ee=0):
    """loss version 4"""

    cumulative_pred = [None] * num_ee + [pred[num_ee]]

    pred_loss = F.mse_loss(cumulative_pred[-1], target)

    return pred_loss, 0, 0


def loss_v5(pred, target, num_ee=0):
    """loss version 5"""

    cumulative_pred = [None] * num_ee + [pred[num_ee]]

    pred_loss = F.smooth_l1_loss(cumulative_pred[-1], target)

    return pred_loss, 0, 0


