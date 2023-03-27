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
        preds, confs = list(), list()

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            pred, conf = exitblock(x)

            if not self.training and conf.item() > self.exit_threshold:
                return pred, idx, self.cost[idx]
            
            preds.append(pred)
            confs.append(conf)

        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        conf = self.confidence(x)

        if not self.training:
            return pred, len(self.exits), 1.0
        
        preds.append(pred)
        confs.append(conf)

        return preds, confs, self.cost