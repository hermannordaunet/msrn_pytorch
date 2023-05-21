import random
import numpy as np

import torch
import torch.optim as optim

# Local imports
from utils.replay_memory import ReplayMemory
from utils.prioritized_memory import PrioritizedMemory
from utils.loss_functions import loss_v1, loss_v2, loss_v3

from utils.print_utils import print_min_max_conf
from utils.data_utils import min_max_conf_from_dataset


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        qnetwork_local,
        qnetwork_target,
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

        self.qnetwork_local = qnetwork_local
        self.qnetwork_target = qnetwork_target

        self.seed = self.model_param["manual_seed"]
        self.learning_rate = self.config["learningRate"]["lr"]
        self.memory_size = self.config["memory_size"]
        self.prioritized_memory = self.config["prioritized_memory"]
        self.batch_size = self.config["batch_size"]

        self.gamma = self.dqn_param["gamma"]
        self.tau = self.dqn_param["tau"]
        self.update_every = self.dqn_param["update_every"]

        self.device = self.model_param["device"]
        # self.small_eps = small_eps # For prioritized memory

        self.cum_loss = None
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

        # If enough samples are available in memory, get random subset and learn
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
        """

        if num_teams > 1:
            # state = torch.from_numpy(state).float().to(self.device)
            state = state.to(
                self.device
            )  # Try to get the state to the same device as model

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values, exits, costs, confs = self.qnetwork_local(state)
            
            # Same for everyone
            laser_action = np.zeros((1, 1))

            move_actions_batch = np.zeros((num_teams, 1, self.qnetwork_local.num_classes))

            if random.random() > eps:
                # Returning action for network
                action_indexes = torch.max(action_values, dim=1)[1]
                # CRITICAL: LOOP - Slow for-loop?
                for count in range(num_teams):
                    move_actions_batch[count, :, action_indexes[count]] = 1.0

            else:
                # Returning a random action
                high = self.qnetwork_local.num_classes
                random_action_idx = np.random.randint(0, high, size=num_teams)
                # CRITICAL: LOOP - Slow for-loop?
                for count in range(num_teams):
                    move_actions_batch[count, :, random_action_idx[count]] = 1.0
            
            return
        
        else:
            # state = torch.from_numpy(state).float().to(self.device)
            state = state.to(
                self.device
            )  # Try to get the state to the same device as model

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values, idx, cost, conf = self.qnetwork_local(state)

            # Epsilon-greedy action selection

            laser_action = np.zeros((1, 1))  # CRITICAL: Get this working
            # Either new network or threshold on the output.

            move_action = np.zeros((1, self.qnetwork_local.num_classes))

            if random.random() > eps:
                # Returning action for network
                action_idx = action_values.max(1)[1].item()
                move_action[0][action_idx] = 1.0
            else:
                # Returning a random action
                high = self.qnetwork_local.num_classes
                random_action_idx = np.random.randint(0, high)
                move_action[0][random_action_idx] = 1.0

            return move_action, laser_action, idx, cost, conf

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        # CRITICAL: Understand all this code. Written for the IN5490 project.
        # I have forgotten all the details of the DQN training loop with the
        # local and tarfet network.

        num_ee = len(self.qnetwork_local.exits)

        # TODO: Check if this is the correct place to start the training
        self.qnetwork_local.train()
        self.qnetwork_target.train()

        self.optimizer.zero_grad()

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
        pred, _, _ = self.qnetwork_target(next_state_batch)
        # CRITICAL: Here we get the Q_targets from the last exit of the network
        Q_targets_next = pred[-1].detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = reward_batch + (self.gamma * Q_targets_next * (1 - dones_batch))

        # ASK: The Q_targets have no "info" of which action it took to get the score

        # Get expected Q values from local model
        pred, conf, cost = self.qnetwork_local(state_batch)
        cost.append(torch.tensor(1.0).to(self.device))

        Q_expected = list()
        for p in pred:
            expected_value = p.gather(1, action_batch)
            Q_expected.append(expected_value)

        cum_loss, pred_loss, cost_loss = loss_v3(
            num_ee, Q_expected, Q_targets, conf, cost
        )

        # Append conf to a list for debugging later
        self.cum_loss = cum_loss
        self.pred_loss = pred_loss
        self.cost_loss = cost_loss
        self.train_conf = conf

        # Minimize the loss
        cum_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def initalize_optimizer(self):
        # Getting the network parameters
        parameters = self.qnetwork_local.parameters()

        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=self.config["learningRate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )

        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(
                parameters,
                lr=self.config["learningRate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                parameters,
                lr=self.config["learningRate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(
                parameters,
                lr=self.config["learningRate"]["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("invalid optimizer")
