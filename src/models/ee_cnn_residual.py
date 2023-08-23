import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import
from src.models.cnn_residual import CNN_Residual
from src.models.utils.exitblock import ExitBlock
from src.models.utils.basicblock import BasicBlock
from src.models.utils.classifier import classifier_linear_softmax, classifier_linear
from src.models.utils.confidence import confidence_linear_sigmoid

from src.models.utils.flops_counter import get_model_complexity_info


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EE_CNN_Residual(nn.Module):
    def __init__(
        self,
        input_shape=(3, 280, 280),
        frames_history=None,
        num_classes=10,
        block=BasicBlock,
        num_ee=1,
        exit_type="bnpool",
        exit_threshold=0.9,
        repetitions=list(),
        init_planes=int(),
        planes=list(),
        distribution=None,
        initalize_parameters=True,
    ):
        super(EE_CNN_Residual, self).__init__()

        # Add depth of the history
        if frames_history:
            input_shape[0] = input_shape[0] * frames_history

        counterpart_model = CNN_Residual(
            input_shape=tuple(input_shape),
            num_classes=num_classes,
            block=block,
            repetitions=repetitions,
            init_planes=init_planes,
            planes=planes,
        )

        self.init_planes = init_planes
        self.planes = planes
        self.input_shape = tuple(input_shape)
        self.channel = self.input_shape[0]
        self.num_classes = num_classes
        self.block = block

        # Create the early exit variables
        self.num_ee = num_ee
        self.exit_type = exit_type
        self.exit_threshold = exit_threshold

        # Containers for the network
        self.layers = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Cost at each exit and the complexity
        self.cost = list()
        self.complexity = list()

        self.stage_id = 0

        # # Get the model just without the ee blocks
        # counterpart_model = CNN_Residual(
        #     input_shape=self.input_shape,
        #     num_classes=self.num_classes,
        #     block=self.block,
        #     repetitions=repetitions,
        #     init_planes=self.init_planes,
        #     planes=self.planes,
        # )

        # Complexity of the entire model and threshold for the early exit
        total_flops, total_params = self.get_complexity(counterpart_model)

        self.set_thresholds(distribution, total_flops)

        # Inital layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.channel,
                    self.init_planes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(self.init_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )

        self.inplanes = self.init_planes
        stride = 1

        for idx, repetition in enumerate(repetitions):
            planes = self.planes[idx]

            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            self.layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            if self.is_suitable_for_exit():
                self.add_exit_block(exit_type, total_flops)
                print(f"Added exit at repetition: {idx+1}, after first block")

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))
                if self.is_suitable_for_exit():
                    self.add_exit_block(exit_type, total_flops)
                    print(f"Added exit at repetition: {idx+1}, after second block")

            # planes = self.planes[idx + 1]
            stride = 2

        self.layers.append(nn.AdaptiveAvgPool2d(1))

        # Dropout layer for generalization and overfitting
        # TODO: Find out if this is nice to have in the CNN
        # self.dropout = nn.Dropout(dropout_prob)

        in_size = planes * block.expansion
        self.classifier = classifier_linear(self.num_classes, in_size)
        self.confidence = confidence_linear_sigmoid(in_size)

        self.stages.append(nn.Sequential(*self.layers))

        # Needs to be here to get the correct cost
        self.complexity.append((total_flops, total_params))

        if initalize_parameters:
            self.parameter_initializer()

    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee + 1)
        self.threshold = list()
        for i in range(self.num_ee):
            if distribution == "pareto":
                self.threshold.append(total_flops * (1 - (0.8 ** (i + 1))))
            elif distribution == "fine":
                self.threshold.append(total_flops * (1 - (0.95 ** (i + 1))))
            elif distribution == "linear":
                self.threshold.append(total_flops * flop_margin * (i + 1))
            else:
                self.threshold.append(total_flops * (gold_rate ** (i - self.num_ee)))

    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(
            model, self.input_shape, print_per_layer_stat=False, as_strings=False
        )
        return flops, params

    def parameter_initializer(self, zero_init_residual=False):
        """
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros,
        and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def is_suitable_for_exit(self):
        """is the position suitable to locate an early-exit block"""
        intermediate_model = nn.Sequential(*(list(self.stages) + list(self.layers)))
        flops, _ = self.get_complexity(intermediate_model)
        return self.stage_id < self.num_ee and flops >= self.threshold[self.stage_id]

    def add_exit_block(self, exit_type, total_flops):
        """add early-exit blocks to the model

        Argument is
        * total_flops:   the total FLOPs of the counterpart model.

        This add exit blocks to suitable intermediate position in the model,
        and calculates the FLOPs and parameters until that exit block.
        These complexity values are saved in the self.cost and self.complexity.
        """

        self.stages.append(nn.Sequential(*self.layers))
        self.exits.append(
            ExitBlock(self.inplanes, self.num_classes, self.input_shape, exit_type)
        )
        intermediate_model = nn.Sequential(*(list(self.stages) + list(self.exits)[-1:]))
        flops, params = self.get_complexity(intermediate_model)
        self.cost.append(flops / total_flops)
        self.complexity.append((flops, params))
        self.layers = nn.ModuleList()
        self.stage_id += 1

    def forward(self, x):
        not_batch_eval = x.shape[0] == 1

        if self.training:
            preds, confs, cost = list(), list(), list()

        if not self.training:
            self.original_idx = None

            self.val_batch_pred = None
            self.val_batch_conf = None
            self.val_batch_exit = None
            self.val_batch_cost = None

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            pred, conf = exitblock(x)

            if not self.training:
                # exit condition:
                if not_batch_eval:
                    conf_over_threshold = conf.item() > self.exit_threshold

                    if conf_over_threshold:
                        return pred, conf.item(), idx, self.cost[idx]

                else:
                    idx_to_remove = self.construct_validation_output(
                        pred,
                        conf,
                        self.cost[idx],
                        idx,
                    )

                    if idx_to_remove is not None:
                        x = remove_exited_pred_from_batch(x, idx_to_remove)

            else:
                preds.append(pred)
                confs.append(conf)
                cost.append(self.cost[idx])

        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        conf = self.confidence(x)

        if not self.training:
            if conf.shape[0] == 1:
                return pred, conf.item(), len(self.exits), 1.0

            self.construct_validation_output(
                pred, conf, 1, len(self.exits), threshold=float("-inf")
            )

            return (
                self.val_batch_pred,
                self.val_batch_conf,
                self.val_batch_exit,
                self.val_batch_cost,
            )

        preds.append(pred)
        confs.append(conf)

        return preds, confs, cost

    def find_conf_above_threshold(self, conf, threshold=None):
        if threshold:
            exit_threshold = threshold
        else:
            exit_threshold = self.exit_threshold

        idx = torch.where(conf > exit_threshold)[0]
        empty = idx.shape[0] == 0

        return idx, empty

    def construct_validation_output(
        self,
        pred,
        conf,
        cost,
        exit_idx,
        threshold=None,
    ):
        if self.val_batch_pred is None:
            self.val_batch_pred = torch.zeros_like(pred)
            self.val_batch_conf = torch.zeros_like(conf)
            self.val_batch_exit = torch.zeros_like(conf, dtype=torch.int)
            self.val_batch_cost = torch.zeros_like(conf)

        if self.original_idx is None:
            self.original_idx = torch.zeros_like(conf, dtype=torch.int64).squeeze()
            self.original_idx[:] = torch.arange(conf.shape[0])

        idx_to_remove, remove_idx_empty = self.find_conf_above_threshold(
            conf, threshold=threshold
        )

        if remove_idx_empty:
            return None

        sample_idx = get_elements_from_indices(self.original_idx, idx_to_remove)

        self.val_batch_pred[sample_idx, :] = pred[idx_to_remove, :]
        self.val_batch_conf[sample_idx, :] = conf[idx_to_remove, :]
        self.val_batch_exit[sample_idx] = exit_idx
        self.val_batch_cost[sample_idx] = cost

        self.original_idx = remove_indices_from_tensor(self.original_idx, idx_to_remove)

        if self.original_idx is not None:
            return idx_to_remove


def remove_indices_from_tensor(tensor, indices):
    # return None if all the elements should be removed
    if len(tensor) == len(indices):
        return None

    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    filtered_tensor = tensor[mask]

    return filtered_tensor


def get_elements_from_indices(tensor, indices):
    # return the original tensor if we ask for all the indices in the tensor.
    if tensor.shape == indices.shape:
        return tensor

    sliced_tensor = tensor[indices]

    return sliced_tensor


def remove_exited_pred_from_batch(x, idx):
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[idx, ...] = False

    # Apply the mask to the tensor
    x_filtered = x[mask]

    # Reshape the filtered tensor to match the original shape
    new_shape = (x.shape[0] - len(idx), x.shape[1], x.shape[2], x.shape[3])
    x_filtered = x_filtered.reshape(new_shape)

    return x_filtered


def main():
    DEVICE = "mps"
    # ee_policy_net = EE_CNN_Residual(
    #     # frames_history=2,
    #     num_ee=1,
    #     planes=[32, 64, 64],
    #     input_shape=(5, 40, 40),
    #     num_classes=3,
    #     repetitions=[2, 2],
    #     distribution="pareto",
    # ).to(DEVICE)

    # ee_target_net = EE_CNN_Residual(
    #     # frames_history=2,
    #     num_ee=1,
    #     planes=[32, 64, 64],
    #     input_shape=(5, 40, 40),
    #     num_classes=3,
    #     repetitions=[2, 2],
    #     distribution="pareto",
    #     initalize_parameters=False,
    # ).to(DEVICE)

    # ee_target_net.load_state_dict(ee_policy_net.state_dict())

    # ee_target_net.eval()
    # ee_policy_net.eval()

    # input = torch.rand(5, 5, 40, 40).to(DEVICE)

    # out_policy = ee_policy_net(input)
    # out_target = ee_target_net(input)

    # print(out_policy == out_target)


if __name__ == "__main__":
    main()
