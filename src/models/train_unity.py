import time
import torch
import numpy as np

import matplotlib.pyplot as plt

from sys import platform

# import the necessary torch packages
from collections import deque

# import the necessary sklearn packages
# from sklearn.metrics import classification_report

# import the necassary scipy packages
# from scipy import stats

# Local import
from utils.agent import Agent
from ee_cnn_residual import EE_CNN_Residual
from utils.data_utils import min_max_conf_from_dataset
from utils.print_utils import print_min_max_conf, print_cost_of_exits

# unity imports
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from utils.stats_side_channel import StatsSideChannel


def get_grid_based_perception_numpy(agent_obs):
    state = agent_obs[0]
    grid_based_perception = np.transpose(state, (2, 0, 1))

    return np.expand_dims(grid_based_perception, axis=0)


def get_grid_based_perception(agent_obs):
    state = agent_obs[0]
    grid_based_perception = torch.tensor(
        state.transpose((2, 0, 1)), dtype=torch.float32
    )

    return grid_based_perception.unsqueeze(0)


def main():
    print(platform)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"[INFO] Device is: {DEVICE}")

    ENV_NAME = "FoodCollector"
    TRAIN_MODEL = True
    VERBOSE = True

    SEED = 1804
    NO_GRAPHICS = True
    FRAME_HISTORY_LEN = 4
    MEMORY_SIZE = int(1e5)
    NUM_EPISODES = 10
    BENCHMARK_MEAN_REWARD = 40

    INIT_LR = 1e-3
    BATCH_SIZE = 64

    # Environment parameters
    LASER_LENGTH = 1.5
    AGENT_SCALE = 1

    # Unity environment spesific
    float_parameter_channel = EnvironmentParametersChannel()
    stats_side_channel = StatsSideChannel()
    SIDE_CHANNELS = [float_parameter_channel, stats_side_channel]

    if platform == "linux" or platform == "linux2":
        relative_path = "builds/Linus_FoodCollector_4_no_respawn.x86_64"
        FILE_NAME = relative_path
    else:
        relative_path = "builds/FoodCollector_4_no_respawn.app"
        FILE_NAME = relative_path

    env = UnityEnvironment(
        file_name=FILE_NAME,
        seed=SEED,
        side_channels=SIDE_CHANNELS,
        no_graphics=NO_GRAPHICS,
    )

    # Unity environment spesific
    float_parameter_channel.set_float_parameter("laser_length", LASER_LENGTH)
    float_parameter_channel.set_float_parameter("agent_scale", AGENT_SCALE)

    env.reset()

    behavior_specs = env.behavior_specs
    behavior_names = env.behavior_specs.keys()
    team_name_list = list(behavior_names)

    action_spec_list = list()
    observation_spec_list = list()
    for team_name in team_name_list:
        action_spec_list.append(behavior_specs[team_name].action_spec)
        observation_spec_list.append(behavior_specs[team_name].observation_specs)

    if len(set(action_spec_list)) == 1:
        action_spec = action_spec_list[-1]
    else:
        print("The agents has differing action specs. Needs to be implemented")
        exit()

    observation_spec = observation_spec_list[0]
    continuous_size = action_spec.continuous_size
    discrete_size = action_spec.discrete_size

    input_size = observation_spec[-1].shape[::-1]
    channels, screen_width, screen_height = input_size

    # TODO: Implement a test to see if the agents has the same observation specs
    # if len(set(observation_spec_list)) == 2:
    #     observation_spec =  observation_spec_list[-1]
    # else:
    #     print("The agents has differing observation specs. Needs to be implemented")
    #     exit()

    if TRAIN_MODEL:
        ee_qnetwork_local = EE_CNN_Residual(
            input_shape=input_size,
            # frames_history=2,
            num_classes=continuous_size,
            num_ee=2,
            repetitions=[2, 2],
            planes=[32, 64, 64],
            distribution="pareto",
        ).to(DEVICE)

        ee_qnetwork_target = EE_CNN_Residual(
            input_shape=input_size,
            # frames_history=2,
            num_classes=continuous_size,
            num_ee=2,
            repetitions=[2, 2],
            planes=[32, 64, 64],
            distribution="pareto",
        ).to(DEVICE)

        ee_qnetwork_local.eval()
        ee_qnetwork_target.eval()

        # TUNE: This object has some init values to look into.
        agent = Agent(
            ee_qnetwork_local,
            ee_qnetwork_target,
            seed=SEED,
            learning_rate=INIT_LR,
            memory_size=MEMORY_SIZE,
            prioritized_memory=False,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )

        startTime = time.time()
        print(f"[INFO] started training @ {time.ctime(startTime)}")

        scores, i, scores_window, losses = model_trainer(
            env,
            agent,
            num_episodes=NUM_EPISODES,
            early_stop=BENCHMARK_MEAN_REWARD,
            verbose=VERBOSE,
        )

        endTime = time.time()
        print(
            "[INFO] total time taken to train the model: {:.2f}s".format(
                endTime - startTime
            )
        )

        # save scores plot
        plt.figure()
        plt.plot(np.arange(len(scores)), scores)
        plt.title(ENV_NAME)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.savefig(f"./reports/figures/{ENV_NAME}_train_scores.png")

        # save loss plot
        plt.figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.title(ENV_NAME)
        plt.ylabel("Train loss")
        plt.xlabel("Episode #")
        plt.savefig(f"./reports/figures/{ENV_NAME}_train_losses.png")

        exit()


def model_trainer(
    env,
    agent,
    num_episodes=100,
    print_range=10,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.95,
    early_stop=13,
    verbose=False,
):
    """Deep Q-Learning trainer.

    Params
    ======
        num_episodes (int): maximum number of training episodes
        print_range (int): range to print partials results
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        early_stop (int): Stop training when achieve a defined score.
    """

    scores = list()  # list containing scores from each episode
    losses = list()
    scores_window = deque(maxlen=print_range)
    eps = eps_start  # initialize epsilon

    # Get the name of the environment object
    env_object = list(env._env_specs)[0]

    # Get a random agent id for training
    decision_steps, _ = env.get_steps(env_object)
    agent_ids = decision_steps.agent_id
    agent_id = agent_ids[0]

    for i in range(1, num_episodes + 1):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(env_object)
        agent_ids = decision_steps.agent_id

        episode_score = 0

        agent_obs = decision_steps[agent_id].obs
        state = get_grid_based_perception(agent_obs)

        min_max_conf = list()

        while True:
            if agent_id in agent_ids:
                act = agent.act(state, eps)
                move_action, laser_action, idx, cost, conf = act

                env.set_action_for_agent(
                    env_object,
                    agent_id,
                    ActionTuple(move_action, laser_action),
                )

            env.step()

            decision_steps, terminal_steps = env.get_steps(env_object)
            agent_ids = decision_steps.agent_id

            # ASK: This needs to be if agent not done?s
            if agent_id in agent_ids:
                agent_obs = decision_steps[agent_id].obs
                next_state = get_grid_based_perception(agent_obs)
            else:
                next_state = None

            reward = decision_steps[agent_id].reward

            terminated_agent_ids = terminal_steps.agent_id
            done = (
                terminal_steps[agent_id].interrupted
                if agent_id in terminated_agent_ids
                else False
            )

            action = np.argmax(move_action)
            optimized = agent.step(state, action, reward, next_state, done, i)
            state = next_state
            episode_score += reward

            if optimized:
                min_max_conf.append(agent.train_conf)

            if done:
                break

        scores_window.append(episode_score)  # save most recent score
        scores.append(episode_score)  # save most recent score
        losses.append(agent.cum_loss)  # save most recent loss

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        if verbose:
            print(f"\rEpisode {i}/{num_episodes}: ")
            print(
                f"Average Score last {len(scores_window)} episodes: {np.mean(scores_window):.2f}"
            )
            print(f"Last loss: {agent.cum_loss}")

            min_vals, max_vals = min_max_conf_from_dataset(min_max_conf)
            print_min_max_conf(min_vals, max_vals)

        if early_stop:
            if np.mean(scores_window) >= early_stop and i > 10:
                if verbose:
                    print(
                        "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                            i, np.mean(scores_window)
                        )
                    )
            break

    return scores, i, scores_window, losses


if __name__ == "__main__":
    main()
