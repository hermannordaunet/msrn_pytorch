import random
import numpy as np

import torch
import torch.optim as optim

# Local imports
from utils.replay_memory import ReplayMemory
from utils.prioritized_memory import PrioritizedMemory
from utils.loss_functions import (
    loss_v1,
    loss_v2,
    loss_v3,
    loss_v4,
    loss_v5,
    loss_v6,
    loss_v7,
    loss_v8,
    loss_exit,
)

from utils.print_utils import print_min_max_conf
from utils.data_utils import min_max_conf_from_dataset

from visualization.visualize import plot_grid_based_perception


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        policy_net,
        target_net,
        model_param=None,
        config=None,
        dqn_param=None,
    ):
        """Initialize an Agent object."""

        if model_param is None:
            print("Cannot initialize agent without model_param dict")
            exit()

        if config is None:
            print("Cannot initialize agent without config dict")
            exit()

        if dqn_param is None:
            print("Cannot initialize agent without dqn_param dict")
            exit()

        self.model_param = model_param
        self.config = config
        self.dqn_param = dqn_param

        self.policy_net = policy_net
        self.target_net = target_net

        self.seed = self.model_param["manual_seed"]
        random.seed(self.seed)

        self.learning_rate = self.config["learning_rate"]["lr"]
        self.memory_size = self.config["memory_size"]
        self.minimal_memory_size = self.config["minimal_memory_size"]
        self.prioritized_memory = self.config["prioritized_memory"]
        self.batch_size = self.config["batch_size"]
        self.multiple_epochs = self.config["multiple_epochs"]
        self.num_epochs = self.config["num_epochs"]
        self.double_dqn = self.config["double_dqn"]

        self.gamma = self.dqn_param["gamma"]
        self.tau = self.dqn_param["tau"]
        self.update_every = self.dqn_param["update_every"]

        self.clip_gradients = config["clip_gradients"]
        if config["max_grad_norm"]:
            self.max_grad_norm = config["max_grad_norm"]
        else:
            self.max_grad_norm = 1

        self.device = self.model_param["device"]
        # self.small_eps = small_eps # For prioritized memory

        self.pred_loss = None
        self.cost_loss = None
        self.train_conf = None
        self.full_net_loss = None
        self.cumulative_exits_loss = None

        if self.prioritized_memory:
            self.memory = PrioritizedMemory(self.memory_size, self.batch_size)
        else:
            self.memory = ReplayMemory(self.memory_size, self.batch_size)

        self.optimizer = None
        self.exit_optimizer = None

        if self.model_param["exit_loss_function"] is None:
            self.initalize_full_optimizer()
        else:
            self.initalize_optimizer()
            self.initalize_exit_optimizer()

        if config["use_lr_scheduler"]:
            use_scheduler_milestones = config["scheduler_milestones"] is not None
            scheduler_factor = config["scheduler_factor"] is not None
            if use_scheduler_milestones and scheduler_factor:
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=config["scheduler_milestones"],
                    gamma=config["scheduler_factor"],
                )
        else:
            self.scheduler = None

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step != 0:
            return False

        if len(self.memory) < self.minimal_memory_size:
            return False

        # If enough samples are available in memory, get random subset and learn
        # Redundant now that we have minimal_memory_size
        if len(self.memory) < self.batch_size:
            return False

        # if self.prioritized_memory:
        #     experiences = self.memory.sample(self.get_beta(i))
        # else:
        #     experiences = self.memory.sample()

        if self.multiple_epochs:
            multiple_experiences = self.memory.sample(
                multiple_experiences=True, number_of_experiences=self.num_epochs
            )
            for epoch in range(self.num_epochs):
                experiences = multiple_experiences[epoch]
                self.old_learn(experiences)
        else:
            experiences = self.memory.sample()
            self.old_learn(experiences)

        return True

    def act(self, act_state, epsilon=0.0, num_agents=1, eval_agent=False, eval_all_exit=False, eval_exit_point=None):
        """Returns actions for given state as per current policy.

        Params:
            act_state (array_like): current state sent to act
            eps (float): epsilon, for epsilon-greedy action selection
            num_teams (int): how many environments in parallell
        """

        # Same for everyone
        laser_action_batch = np.zeros((num_agents, 1, 1))
        move_actions_batch = np.zeros((num_agents, 1, self.policy_net.num_classes))

        random_number = random.random()
        if epsilon <= random_number:
            if eval_agent:
                self.policy_net.forced_exit_point = None
            else:
                self.policy_net.forced_exit_point = self.policy_net.num_ee + 1
            
            if eval_all_exit:
                self.policy_net.forced_exit_point = eval_exit_point

            # Returning action from the last exit of the network
            act_state = act_state.to(self.device)

            was_in_training = self.policy_net.training

            self.policy_net.eval()

            with torch.no_grad():
                # action_values, confs, exits, costs = self.policy_net(act_state)
                action_values, confs, exits, costs = self.policy_net(act_state)

            _, action_indexes = torch.max(action_values, dim=1)

            # CRITICAL: Slow for-loop?
            for count in range(num_agents):
                move_actions_batch[count, :, action_indexes[count]] = 1.0

            if was_in_training:
                self.policy_net.train()

            self.policy_net.forced_exit_point = None

        else:
            # Returning a random action
            high = self.policy_net.num_classes
            random_action_idx = np.random.randint(0, high, size=num_agents)
            # CRITICAL: Slow for-loop?
            for count in range(num_agents):
                move_actions_batch[count, :, random_action_idx[count]] = 1.0

            exits = None
            costs = None
            confs = None

        return (
            move_actions_batch,
            laser_action_batch,
            confs,
            exits,
            costs,
        )  # confs, exits, costs

    def old_learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        # CRITICAL: Understand all this code. Written for the IN5490 project.
        # I have forgotten all the details of the DQN training loop with the
        # local and tarfet network.

        num_ee = len(self.policy_net.exits)

        batch = self.memory.Transition(*zip(*experiences))

        # Adding all the variables to the device?
        # TODO: does all the things needs to
        # be on the device (MPS/GPU). Most likely only the state and next-state
        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = (
            torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        dones_batch = (
            torch.tensor(batch.done, dtype=torch.int).unsqueeze(1).to(self.device)
        )

        # plot_grid_based_perception(state_batch[0:9, ...], title="10 first states", block=False)
        # plot_grid_based_perception(next_state_batch[0:9, ...], title="10 first next states", block=False)

        # print(action_batch[0:9, ...])
        # print(dones_batch[0:9, ...])
        # print(reward_batch[0:9, ...])

        if self.target_net:
            # Get max predicted Q values (for next states) from target model
            next_pred, _, _ = self.target_net(next_state_batch)

        else:
            print("[ERROR] The agent has no target net. Only use for eval/visualize")
            exit()

        # CRITICAL: Here we get the Q_targets from the last exit of the network
        # Here we need the network to be set up with some kind of inf threshold
        # to get the prediction from the last exit
        Q_targets_next = next_pred[-1].detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = reward_batch + (self.gamma * Q_targets_next * (1 - dones_batch))

        # ASK: The Q_targets have no "info" of which action it took to get the score

        self.policy_net.forced_exit_point = None
        # Get expected Q values from policy model
        pred, conf, cost = self.policy_net(state_batch)
        # CRITICAL: Add back the correct loss
        # cost.append(torch.tensor(1.0).to(self.device))

        Q_expected = list()
        for p in pred:
            expected_value = p.gather(1, action_batch)
            Q_expected.append(expected_value)

        loss = self.initalize_loss_function()

        cumulative_loss, pred_loss, cost_loss = loss(
            Q_expected, Q_targets, conf, cost, num_ee=num_ee
        )

        # Append conf to a list for debugging later
        self.full_net_loss = cumulative_loss
        self.pred_loss = pred_loss
        self.cost_loss = cost_loss
        self.train_conf = conf
        self.last_Q_expected = Q_expected[-1]
        self.last_Q_targets = Q_targets

        # Minimize the loss
        self.optimizer.zero_grad()
        cumulative_loss.backward()

        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        # CRITICAL: Understand all this code. Written for the IN5490 project.
        # I have forgotten all the details of the DQN training loop with the
        # local and tarfet network.

        # self.freeze_exit_layers()

        num_ee = len(self.policy_net.exits)

        batch = self.memory.Transition(*zip(*experiences))

        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = (
            torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        dones_batch = (
            torch.tensor(batch.done, dtype=torch.int).unsqueeze(1).to(self.device)
        )

        # plot_grid_based_perception(state_batch[0:9, ...], title="10 first states", block=False)
        # plot_grid_based_perception(next_state_batch[0:9, ...], title="10 first next states", block=False)

        # print(action_batch[0:9, ...])
        # print(dones_batch[0:9, ...])
        # print(reward_batch[0:9, ...])

        if self.target_net:
            # Get max predicted Q values (for next states) from target model
            next_pred, _, _ = self.target_net(next_state_batch)

        else:
            print("[ERROR] The agent has no target net. Only use for eval/visualize")
            exit()

        # -------------------------------------------------------------------------------------- 
        # Double DQN
        if self.double_dqn:

            # TODO: Check if this can be used to make Double DQN
            # Use policy_net to select the action that maximizes Q-values for the next state
            next_action, _, _ = self.policy_net(next_state_batch)
            max_next_action = torch.argmax(next_action[-1].detach(), dim=1)

            # Use target_net to evaluate the Q-value of taking that action in the next state
            Q_values_next_state = next_pred[-1].detach().gather(1, max_next_action.unsqueeze(1))

            # Compute Q-targets using the reward and discounted Q-values of the next state
            Q_targets = reward_batch + (self.gamma * Q_values_next_state * (1 - dones_batch))

        # -------------------------------------------------------------------------------------- 
        else:
        # -------------------------------------------------------------------------------------- 
            # DQN

            Q_targets_next = next_pred[-1].detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            Q_targets = reward_batch + (self.gamma * Q_targets_next * (1 - dones_batch))

        # ASK: The Q_targets have no "info" of which action it took to get the score
        # -------------------------------------------------------------------------------------- 
        

        self.policy_net.forced_exit_point = None
        # Get expected Q values from policy model
        pred, conf, cost = self.policy_net(state_batch)
        # CRITICAL: Add back the correct loss
        # cost.append(torch.tensor(1.0).to(self.device))

        # Q_expected = list()
        # for p in pred:
        #     expected_value = pred[-1].gather(1, action_batch)
        #     Q_expected.append(expected_value)

        loss = self.initalize_loss_function()
        exit_loss = self.initalize_exit_loss()

        Q_expected = pred[-1].gather(1, action_batch)
        q_full_net_loss = loss(Q_expected, Q_targets, num_ee=num_ee)

        # q_full_net_loss, pred_loss_exits, cumulative_loss, cost_loss, last_Q_expected = loss(
        #     pred, Q_targets, action_batch, cost, num_ee=num_ee
        # )

        conf = list()
        for p in pred:
            sliced_pred = p[torch.arange(self.batch_size), action_batch.squeeze()]
            conf.append(sliced_pred.unsqueeze(1))

        # conf = torch.max(conf_list, 1)[0]
        # Append conf to a list for debugging later
        self.train_conf = conf
        # self.cost_loss = cost_loss
        self.pred_loss = q_full_net_loss

        self.last_Q_targets = Q_targets
        self.last_Q_expected = Q_expected[-1]

        self.full_net_loss = q_full_net_loss

        # Minimize the loss
        self.optimizer.zero_grad()
        q_full_net_loss.backward()

        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        pred, conf, cost = self.policy_net(state_batch)

        loss_exit = None
        exit_preds = pred[:-1]
        exit_costs = cost[:-1]
        for idx, (exit_pred, exit_cost) in enumerate(zip(exit_preds, exit_costs)):
            if loss_exit is None:
                loss_exit = exit_loss(exit_pred, exit_cost, action_batch)
            else:
                loss_exit += exit_loss(exit_pred, exit_cost, action_batch)

        self.cumulative_exits_loss = loss_exit

        self.exit_optimizer.zero_grad()
        loss_exit.backward()

        self.exit_optimizer.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, policy_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
            policy_net (PyTorch model): weights will be copied from
            target_net (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_net_param, policy_net_param in zip(
            target_net.parameters(), policy_net.parameters()
        ):
            target_net_param.data.copy_(
                tau * policy_net_param.data + (1.0 - tau) * target_net_param.data
            )

    def freeze_exit_layers(self):
        for exit in self.policy_net.exits:
            exit.classifier[0].require_grad = False

    def unfreeze_exit_layers(self):
        for exit in self.policy_net.exits:
            exit.classifier[0].require_grad = True
        
    def initalize_full_optimizer(self):
        # Getting the network parameters
        policy_net_parameters = self.policy_net.parameters()
        lr_backbone = self.config["learning_rate"]["lr"]

        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )

        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("invalid optimizer")
        

    def initalize_optimizer(self):
        # Getting the network parameters
        policy_net_parameters = self.policy_net.layers_without_exit.parameters()
        lr_backbone = self.config["learning_rate"]["lr"]

        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )

        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(
                policy_net_parameters,
                lr=lr_backbone,
                # weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("invalid optimizer")

    def initalize_exit_optimizer(self):
        exit_parameters = self.policy_net.exits.parameters()
        exit_lr = self.config["learning_rate"]["lr_exit"]

        if self.config["optimizer"] == "adam":
            self.exit_optimizer = optim.Adam(
                exit_parameters,
                lr=exit_lr,
                # weight_decay=self.config["weight_decay"],
            )

        elif self.config["optimizer"] == "adamW":
            self.exit_optimizer = optim.AdamW(
                exit_parameters,
                lr=exit_lr,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            self.exit_optimizer = optim.SGD(
                exit_parameters,
                lr=exit_lr,
                # weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSprop":
            self.exit_optimizer = optim.RMSprop(
                exit_parameters,
                lr=exit_lr,
                # weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("invalid optimizer")

    def initalize_loss_function(self):
        if self.model_param["loss_function"] == "v1":
            return loss_v1
        elif self.model_param["loss_function"] == "v2":
            return loss_v2
        elif self.model_param["loss_function"] == "v3":
            return loss_v3
        elif self.model_param["loss_function"] == "v4":
            return loss_v4
        elif self.model_param["loss_function"] == "v5":
            return loss_v5
        elif self.model_param["loss_function"] == "v6":
            return loss_v6
        elif self.model_param["loss_function"] == "v7":
            return loss_v7
        elif self.model_param["loss_function"] == "v8":
            return loss_v8
        else:
            raise Exception("invalid loss function")

    def initalize_exit_loss(self):
        if self.model_param["exit_loss_function"] == "loss_exit":
            return loss_exit
        elif self.model_param["exit_loss_function"] is None:
            print("Training without dedicated loss exit")
        else:
            raise Exception("invalid loss function for exit nodes")
