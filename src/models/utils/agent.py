import random
import numpy as np

import torch
import torch.optim as optim

# Local imports
from utils.replay_memory import ReplayMemory
from utils.prioritized_memory import PrioritizedMemory
from utils.loss_functions import loss_v1, loss_v2, loss_v3, loss_v4

from utils.print_utils import print_min_max_conf
from utils.data_utils import min_max_conf_from_dataset


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        policy_net,
        target_net,
        model_param=None,
        config=None,
        dqn_param=None,
        # small_eps=1e-5, # For prioritized memory
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            learning_rate (float): Pending
            memory_size (int): Pending
            prioritized_memory (bool): Pending
            batch_size (int): Pending
            gamma (float): Pending
            tau (float): Pending
            small_eps (float): Pending
            update_every (int): Pending
        """

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
        self.learning_rate = self.config["learning_rate"]["lr"]
        self.memory_size = self.config["memory_size"]
        self.minimal_memory_size = self.config["minimal_memory_size"]
        self.prioritized_memory = self.config["prioritized_memory"]
        self.batch_size = self.config["batch_size"]

        self.gamma = self.dqn_param["gamma"]
        self.tau = self.dqn_param["tau"]
        self.update_every = self.dqn_param["update_every"]

        self.device = self.model_param["device"]
        # self.small_eps = small_eps # For prioritized memory

        self.cumulative_loss = None
        self.pred_loss = None
        self.cost_loss = None
        self.train_conf = None

        if self.prioritized_memory:
            self.memory = PrioritizedMemory(self.memory_size, self.batch_size)
        else:
            self.memory = ReplayMemory(self.memory_size, self.batch_size)

        # self.optimizer = None
        self.initalize_optimizer()

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

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, i):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step != 0:
            return False

        # TODO: Add a minimal batch size?
        if len(self.memory) <= self.minimal_memory_size:
            return False

        # If enough samples are available in memory, get random subset and learn
        # Redundant now that we have minimal_memory_size
        if len(self.memory) <= self.batch_size:
            return False

        if self.prioritized_memory:
            experiences = self.memory.sample(self.get_beta(i))
        else:
            experiences = self.memory.sample()

        self.learn(experiences)

        return True

    def act(self, state, eps=0.0, num_teams=1):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            num_teams (int): how many environments in parallell
        """

        # Same for everyone
        laser_action_batch = np.zeros((num_teams, 1, 1))

        move_actions_batch = np.zeros((num_teams, 1, self.policy_net.num_classes))

        if random.random() > eps:
            # Returning action for network
            # action_indexes = torch.max(action_values, dim=1)[1]
            state = state.to(
                self.device
            )  # Try to get the state to the same device as model

            self.policy_net.eval()
            with torch.no_grad():
                action_values, confs, exits, costs = self.policy_net(state)

            action_indexes = torch.argmax(action_values, dim=1)
            # CRITICAL: Slow for-loop?
            for count in range(num_teams):
                move_actions_batch[count, :, action_indexes[count]] = 1.0

        else:
            # Returning a random action
            high = self.policy_net.num_classes
            random_action_idx = np.random.randint(0, high, size=num_teams)
            # CRITICAL: Slow for-loop?
            for count in range(num_teams):
                move_actions_batch[count, :, random_action_idx[count]] = 1.0

        return move_actions_batch, laser_action_batch  # , exits, costs, confs

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        # CRITICAL: Understand all this code. Written for the IN5490 project.
        # I have forgotten all the details of the DQN training loop with the
        # local and tarfet network.

        num_ee = len(self.policy_net.exits)

        # TODO: Check if this is the correct place to start the training
        self.policy_net.train()

        # TODO: Find out if this needs to be in eval or train?
        # self.target_net.train()

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

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            pred, _, _, _ = self.target_net(next_state_batch)

        # CRITICAL: Here we get the Q_targets from the last exit of the network
        # Here we need the network to be set up with some kind of inf threshold
        # to get the prediction from the last exit
        Q_targets_next = pred.detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = reward_batch + (self.gamma * Q_targets_next * (1 - dones_batch))

        # ASK: The Q_targets have no "info" of which action it took to get the score

        # Get expected Q values from policy model
        pred, conf, cost = self.policy_net(state_batch)
        cost.append(torch.tensor(1.0).to(self.device))

        Q_expected = list()
        for p in pred:
            expected_value = p.gather(1, action_batch)
            Q_expected.append(expected_value)

        # cumulative_loss, pred_loss, cost_loss = loss_v2(
        #    Q_expected, Q_targets, conf, cost, num_ee=num_ee
        # )

        cumulative_loss, pred_loss, cost_loss = loss_v4(
            Q_expected, Q_targets, num_ee=num_ee
        )

        # Append conf to a list for debugging later
        self.cumulative_loss = cumulative_loss
        self.pred_loss = pred_loss
        self.cost_loss = cost_loss
        self.train_conf = conf

        # Minimize the loss
        self.optimizer.zero_grad()
        cumulative_loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, policy_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            policy_net (PyTorch model): weights will be copied from
            target_net (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # for target_param, local_param in zip(
        #     target_model.parameters(), local_model.parameters()
        # ):
        #     target_param.data.copy_(
        #         tau * local_param.data + (1.0 - tau) * target_param.data
        #     )

        target_model_state_dict = target_net.state_dict()
        policy_model_state_dict = policy_net.state_dict()
        for key in policy_model_state_dict:
            target_model_state_dict[key] = policy_model_state_dict[
                key
            ] * tau + target_model_state_dict[key] * (1 - tau)

    def initalize_optimizer(self):
        # Getting the network parameters
        parameters = self.policy_net.parameters()

        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=self.config["learning_rate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )

        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(
                parameters,
                lr=self.config["learning_rate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                parameters,
                lr=self.config["learning_rate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(
                parameters,
                lr=self.config["learning_rate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("invalid optimizer")
