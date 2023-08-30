import random
import numpy as np

import torch
import torch.optim as optim

# Local imports
from utils.replay_memory import ReplayMemoryPPO
from utils.prioritized_memory import PrioritizedMemory
from utils.loss_functions import loss_v1, loss_v2, loss_v3, loss_v4, loss_v5

from utils.print_utils import print_min_max_conf
from utils.data_utils import min_max_conf_from_dataset

from visualization.visualize import plot_grid_based_perception


class PPO_Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        policy_net,
        old_policy_net,
        critic_net,
        model_param=None,
        config=None,
        dqn_param=None,
        # small_eps=1e-5, # For prioritized memory
    ):
        """Initialize an Agent object with PPO training."""

        self.model_param = model_param
        self.config = config
        self.dqn_param = dqn_param

        self.policy_net = policy_net
        self.old_policy_net = old_policy_net
        self.critic_net = critic_net

        self.seed = self.model_param["manual_seed"]
        self.learning_rate = self.config["learning_rate"]["lr"]
        self.memory_size = self.config["memory_size"]
        self.minimal_memory_size = self.config["minimal_memory_size"]
        self.prioritized_memory = self.config["prioritized_memory"]
        self.batch_size = self.config["batch_size"]

        self.gamma = self.dqn_param["gamma"]
        self.tau = self.dqn_param["tau"]
        self.update_every = self.dqn_param["update_every"]

        self.device = self.model_param["device"]

        self.cumulative_loss = None
        self.pred_loss = None
        self.cost_loss = None
        self.train_conf = None

        if self.prioritized_memory:
            print(
                "[ERROR]: Prioritized memory not implemented for PPO agent. Using regular replay memory"
            )

        self.memory = ReplayMemoryPPO(self.memory_size, self.batch_size)

        # Optimizer initalize
        self.initalize_optimizer()

        # Loss initalize
        self.initalize_loss_function()

        if config["use_lr_scheduler"]:
            if (config["scheduler_milestones"] is not None) and (
                config["scheduler_factor"] is not None
            ):
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=config["scheduler_milestones"],
                    gamma=config["scheduler_factor"],
                )
        else:
            self.scheduler = None

        self.t_step = 0

    def step(self, state, action, log_prob, reward, done):
        # Save experience in replay memory
        self.memory.add(state, action, log_prob, reward, done)

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

        experiences = self.memory.sample()

        self.learn(experiences)

        return True

    def act(self, act_state, epsilon=0.0, num_agents=1):
        """Returns actions for given state as per current policy.

        Params:
            act_state (array_like): current state sent to act
            eps (float): epsilon, for epsilon-greedy action selection
            num_teams (int): how many environments in parallell
        """

        # Same for everyone
        laser_action_batch = np.zeros((num_agents, 1, 1))

        move_actions_batch = np.zeros((num_agents, 1, self.policy_net.num_classes))

        if random.random() >= epsilon:
            # Returning action for network
            # action_indexes = torch.max(action_values, dim=1)[1]
            act_state = act_state.to(self.device)

            was_in_training = self.policy_net.training

            self.policy_net.eval()
            with torch.no_grad():
                action_values, confs, exits, costs = self.policy_net(act_state)

            action_indexes = torch.argmax(action_values, dim=1)
            # CRITICAL: Slow for-loop?
            for count in range(num_agents):
                move_actions_batch[count, :, action_indexes[count]] = 1.0

            if was_in_training:
                self.policy_net.train()

        else:
            # Returning a random action
            high = self.policy_net.num_classes
            random_action_idx = np.random.randint(0, high, size=num_agents)
            # CRITICAL: Slow for-loop?
            for count in range(num_agents):
                move_actions_batch[count, :, random_action_idx[count]] = 1.0

        return move_actions_batch, laser_action_batch  # , exits, costs, confs

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

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

    def initalize_optimizer(self):
        # Getting the network parameters
        policy_net_parameters = self.policy_net.parameters()
        critic_net_parameters = self.critic_net.parameters()

        lr_actor = self.config["learning_rate"]["lr"]
        lr_critic = self.config["learning_rate"]["lr_critic"]

        weight_decay = self.config["weight_decay"]

        optimizer_input_both_networks = [
            {
                "params": policy_net_parameters,
                "lr": lr_actor,
                "weight_decay": weight_decay,
            },
            {
                "params": critic_net_parameters,
                "lr": lr_critic,
                "weight_decay": weight_decay,
            },
        ]

        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(optimizer_input_both_networks)
        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(optimizer_input_both_networks)
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(optimizer_input_both_networks)
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(optimizer_input_both_networks)
        else:
            raise Exception("invalid optimizer")

    def initalize_loss_function(self):
        if self.model_param["loss_function"] == "v1":
            self.loss = loss_v1
        elif self.model_param["loss_function"] == "v2":
            self.loss = loss_v2
        elif self.model_param["loss_function"] == "v3":
            self.loss = loss_v3
        elif self.model_param["loss_function"] == "v4":
            self.loss = loss_v4
        elif self.model_param["loss_function"] == "v5":
            self.loss = loss_v5
        else:
            raise Exception("invalid loss function")
