import os
import time
import json
import wandb
import torch

import numpy as np

from sys import platform
from pathlib import Path
from collections import deque

# Local import
from utils.agent import Agent
from utils.save_utils import save_model, save_dict_to_json
from ee_cnn_residual import EE_CNN_Residual
from small_dqn import small_DQN
from utils.data_utils import min_max_conf_from_dataset
from utils.print_utils import print_min_max_conf, print_cost_of_exits, get_time_hh_mm_ss
from visualization.visualize import (
    plot_scores_from_list,
    plot_loss_from_list,
    plot_grid_based_perception,
    plot_scores_from_nested_list,
)

# unity imports
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from utils.stats_side_channel import StatsSideChannel


def get_grid_based_perception_numpy(agent_obs):
    grid_state_numpy = agent_obs[0]
    grid_based_perception = np.transpose(grid_state_numpy, (2, 0, 1))

    return np.expand_dims(grid_based_perception, axis=0)


def get_grid_based_perception(agent_obs):
    grid_state = agent_obs[0]
    grid_based_perception = torch.tensor(
        grid_state.transpose((2, 0, 1)), dtype=torch.float32
    )

    return grid_based_perception.unsqueeze(0)


def get_latest_folder(runs_directory: Path):
    # Get all subdirectories in the runs_directory
    subdirectories = [d for d in runs_directory.iterdir() if d.is_dir()]

    # Filter out non-numeric folder names
    subdirectories = [d for d in subdirectories if d.name.isnumeric()]

    if not subdirectories:
        return None

    # Get the folder with the highest Unix timestamp
    latest_folder = max(subdirectories, key=lambda d: int(d.name))
    timestamp = latest_folder.stem

    return latest_folder, timestamp


def load_json_as_dict(file_path):
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"[INFO] Device is: {DEVICE}")

    model_param = {
        "model_class_name": "EE_CNN_Residual",
        "loss_function": "v4",
        "num_ee": 0,
        "repetitions": [2, 2],
        "init_planes": 64,
        "planes": [64, 32],
        "distribution": "pareto",
        # "numbOfCPUThreadsUsed": 10,  # Number of cpu threads use in the dataloader
        "models_dir": None,
        "mode_setups": {"train": True, "val": False, "visualize": False},
        "manual_seed": 1804,  # TODO: Seed everything
        "device": DEVICE,
    }

    config = {
        "env_name": "FoodCollector",
        "use_build": True,
        "no_graphics": True,
        "laser_length": 1.5,
        "agent_scale": 1,
        "prioritized_memory": False,
        "memory_size": 25_000,  # 10_000
        "minimal_memory_size": 999,  # Either batch_size or minimal_memory_size before training
        "batch_size": 512,  # Training batch size
        "num_episodes": 500,
        "benchmarks_mean_reward": None,
        "optimizer": "adam",  # 'SGD' | 'adam' | 'RMSprop' | 'adamW'
        "learning_rate": {"lr": 0.001},  # learning rate to the optimizer
        "weight_decay": 0.00001,  # weight_decay value # TUNE: originally 0.00001
        "use_lr_scheduler": False,
        "scheduler_milestones": [75, 200],  # 45,70 end at 80? or 60, 80
        "scheduler_factor": 0.1,
        "print_range": 10,
        "visualize": {
            "episodes": 10,
        },
    }

    dqn_param = {
        "gamma": 0.999,  # Original: 0.99,
        "tau": 0.005,  # TODO: Try one more 0.05 (5e-2) previous
        "update_every": 10,
    }

    epsilon_greedy_param = {
        "eps_start": 0.9,
        "eps_end": 0.05,
        "eps_decay": 0.95,
    }

    TRAIN_MODEL = model_param["mode_setups"]["train"]
    VISUALIZE_MODEL = model_param["mode_setups"]["visualize"]
    VAL_MODEL = model_param["mode_setups"]["val"]

    TIMESTAMP = None

    VERBOSE = True

    FRAME_HISTORY_LEN = 4  # TODO: Should I add this?

    # Unity environment spesific
    float_parameter_channel = EnvironmentParametersChannel()
    stats_side_channel = StatsSideChannel()

    engine_config_channel = EngineConfigurationChannel()
    engine_config_channel.set_configuration_parameters(time_scale=10)

    SIDE_CHANNELS = [engine_config_channel, float_parameter_channel, stats_side_channel]

    if config["use_build"]:
        if platform == "linux" or platform == "linux2":
            relative_path = (
                "builds/Linus_FoodCollector_1_env_no_respawn_headless.x86_64"
            )
            FILE_NAME = relative_path
        else:
            relative_path = "builds/FoodCollector_1_env_no_respawn.app"
            FILE_NAME = relative_path
    else:
        FILE_NAME = None

    env = UnityEnvironment(
        file_name=FILE_NAME,
        side_channels=SIDE_CHANNELS,
        seed=model_param["manual_seed"],
        no_graphics=config["no_graphics"],
        worker_id=2,
    )

    # Unity environment spesific
    float_parameter_channel.set_float_parameter("laser_length", config["laser_length"])
    float_parameter_channel.set_float_parameter("agent_scale", config["agent_scale"])

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
    # discrete_size = action_spec.discrete_size

    model_param["input_size"] = observation_spec[-1].shape[::-1]
    model_param["num_classes"] = continuous_size
    # channels, screen_width, screen_height = input_size

    # TODO: Implement a test to see if the agents has the same observation specs
    # if len(set(observation_spec_list)) == 2:
    #     observation_spec =  observation_spec_list[-1]
    # else:
    #     print("The agents has differing observation specs. Needs to be implemented")
    #     exit()

    model_type = globals()[model_param["model_class_name"]]

    if DEVICE is not "mps":
        run_wandb = wandb.init(
            project="Master-thesis",
            config={**model_param, **config, **dqn_param, **epsilon_greedy_param},
        )
    else:
        run_wandb = None

    if TRAIN_MODEL:
        timestamp = int(time.time())

        results_directory = Path(f"./results/{timestamp}/")
        parameter_directory = results_directory / "parameters"

        # Check if the directory exists
        if not results_directory.exists():
            # If it doesn't exist, create it
            results_directory.mkdir(parents=True)
            models_directory = results_directory / "models"
            models_directory.mkdir()

            model_param["models_dir"] = f"./{models_directory}"

        if not parameter_directory.exists():
            parameter_directory.mkdir()

            model_param["parameter_dir"] = f"./{parameter_directory}"

        # TODO: Save here? Or later?
        # save_dict_to_json(agent.model_param, f"{results_directory}_model_param.json")
        # save_dict_to_json(agent.config, f"{results_directory}_config.json")
        # save_dict_to_json(agent.dqn_param, f"{results_directory}_dqn_param.json")

        print(f"[INFO] Initalizing Q network policy of type {model_type}")
        ee_policy_net = model_type(
            # frames_history=2,
            num_ee=model_param["num_ee"],
            init_planes=model_param["init_planes"],
            planes=model_param["planes"],
            input_shape=model_param["input_size"],
            num_classes=model_param["num_classes"],
            repetitions=model_param["repetitions"],
            distribution=model_param["distribution"],
        ).to(DEVICE)

        print(f"[INFO] Initalizing Q network target of type {model_type}")
        ee_target_net = model_type(
            # frames_history=2,
            num_ee=model_param["num_ee"],
            init_planes=model_param["init_planes"],
            planes=model_param["planes"],
            input_shape=model_param["input_size"],
            num_classes=model_param["num_classes"],
            repetitions=model_param["repetitions"],
            distribution=model_param["distribution"],
            initalize_parameters=False,
        ).to(DEVICE)

        # TODO: This is important to get the networks initalized with the same weigths
        print("[INFO] Copying weight from target net to policy net")
        ee_target_net.load_state_dict(ee_policy_net.state_dict())

        ee_policy_net.eval()

        # TODO: Find out if this needs to be in eval or train?
        # To ensure the network always is in eval we need to
        # specify it right after init
        ee_target_net.eval()

        print("[INFO] Initalizing a Agent object")
        agent = Agent(
            ee_policy_net,
            ee_target_net,
            model_param=model_param,
            config=config,
            dqn_param=dqn_param,
        )

        startTime = time.time()
        print(f"[INFO] started training @ {time.ctime(startTime)}")

        save_dict_to_json(model_param, f"./{parameter_directory}/model_param.json")
        save_dict_to_json(config, f"./{parameter_directory}/config.json")
        save_dict_to_json(dqn_param, f"./{parameter_directory}/dqn_param.json")
        save_dict_to_json(
            epsilon_greedy_param, f"./{parameter_directory}/epsilon_greedy_param.json"
        )

        if run_wandb:
            run_wandb.watch(ee_policy_net, log_freq=int(5))

        scores, episode, scores_window, losses = model_trainer(
            env,
            agent,
            config=config,
            epsilon_greedy_param=epsilon_greedy_param,
            results_directory=results_directory,
            verbose=VERBOSE,
            wandb=run_wandb,
        )

        endTime = time.time()
        total_sec = endTime - startTime
        print(
            f"[INFO] total time taken to train the model: {get_time_hh_mm_ss(total_sec)} sec"
        )

    if VISUALIZE_MODEL:
        # Close old env and start fresh
        env.close()

        env = UnityEnvironment(
            file_name=FILE_NAME,
            side_channels=SIDE_CHANNELS,
            seed=model_param["manual_seed"],
            no_graphics=False,
        )

        env.reset()
        # If we just trained a model, load that one.
        # If not, load the model from the timestamp provided.

        runs_directory = Path(f"./results/")

        if TRAIN_MODEL:
            results_directory, timestamp = get_latest_folder(runs_directory)
            if not results_directory:
                print("No folders found in the runs_directory.")
                exit()

        else:
            if TIMESTAMP:
                results_directory = runs_directory / str(TIMESTAMP)
            else:
                print("No timestamp provided. Cannot load model without it.")
                exit()

        parameter_directory = results_directory / "parameters"

        model_param = load_json_as_dict(f"{parameter_directory}/model_param.json")
        config = load_json_as_dict(f"{parameter_directory}/config.json")
        dqn_param = load_json_as_dict(f"{parameter_directory}/dqn_param.json")

        print("[INFO] Loading the trained policy net")
        # ee_policy_net = EE_CNN_Residual(
        #     # frames_history=2,
        #     num_ee=model_param["num_ee"],
        #     planes=model_param["planes"],
        #     input_shape=model_param["input_size"],
        #     num_classes=model_param["num_classes"],
        #     repetitions=model_param["repetitions"],
        #     distribution=model_param["distribution"],
        # ).to(DEVICE)

        ee_policy_net = model_type(
            input_shape=model_param["input_size"],
            num_classes=model_param["num_classes"],
        )  # .to(DEVICE)

        models_directory = model_param["models_dir"]
        model_file = f"{models_directory}last_model.pt"
        print(f"[INFO] Loading weights from {model_file}")

        # Override trained device:
        print("[INFO] Overriding the device used for training")
        model_param["device"] = DEVICE

        if ee_policy_net.__class__.__name__ == model_param["model_class_name"]:
            ee_policy_net.load_state_dict(torch.load(model_file, map_location=DEVICE))
            ee_policy_net.to(DEVICE)
        else:
            print(
                "[EXIT] The network you want to load is not the same type as the declaired."
            )
            exit()

        ee_target_net = None

        # Setting the network in evaluation mode
        ee_policy_net.eval()

        print("[INFO] Initalizing a Agent object")
        agent = Agent(
            ee_policy_net,
            ee_target_net,
            model_param=model_param,
            config=config,
            dqn_param=dqn_param,
        )

        visualize_trained_model(env, agent, config, VERBOSE)


def model_trainer(
    env,
    agent,
    config=None,
    epsilon_greedy_param=None,
    results_directory="./",
    wandb=None,
    verbose=False,
):
    """Deep Q-Learning trainer.

    Params
    ======
        print_range (int): range to print partials results
        epsilon_greedy_param (dict): all the epsilon greedy parameters
        early_stop (int): Stop training when achieve a defined score.

    """

    # initialize epsilon greedy param
    eps_start = epsilon_greedy_param["eps_start"]
    eps_end = epsilon_greedy_param["eps_end"]
    eps_decay = epsilon_greedy_param["eps_decay"]
    eps = eps_start

    print_range = config["print_range"]
    early_stop = config["benchmarks_mean_reward"]

    num_episodes = config["num_episodes"]

    scores, losses = list(), list()  # list containing scores/losses from each episode
    scores_window = deque(maxlen=print_range)

    try:
        training_agents = dict()
        team_name_list = list(env.behavior_specs.keys())
        num_teams = len(team_name_list)

        if verbose:
            print(
                f"[INFO] Number of parallell environments during training: {num_teams}"
            )

        state_size = agent.model_param["input_size"]
        state_batch_tensor = torch.zeros((num_teams, *state_size))

        for episode in range(1, num_episodes + 1):
            if verbose:
                print(f"\nEpisode {episode}/{num_episodes} started")

            env.reset()  # TODO: Test with and without this

            for team_idx, team in enumerate(team_name_list):
                decision_steps, _ = env.get_steps(team)
                training_agents[team] = {
                    "agent_id": decision_steps.agent_id[
                        -1
                    ],  # random agent from each team
                    "episode_score": 0,
                }
                agent_id = training_agents[team]["agent_id"]
                agent_obs = decision_steps[agent_id].obs
                state = get_grid_based_perception(agent_obs).detach().clone()
                state_batch_tensor[team_idx, ...] = state

            # min_max_conf = list()
            episode_done = False
            while not episode_done:
                move_action, laser_action = agent.act(
                    state_batch_tensor, epsilon=eps, num_agents=num_teams
                )

                # move_action, laser_action = act  # , idx, cost, conf = act

                for team_idx, team in enumerate(team_name_list):
                    agent_id = training_agents[team]["agent_id"]
                    team_move_action = move_action[team_idx, ...]
                    team_laser_action = laser_action[team_idx, ...]
                    env.set_action_for_agent(
                        team,
                        agent_id,
                        ActionTuple(team_move_action, team_laser_action),
                    )

                env.step()

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, terminal_steps = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    agent_id = training_agents[team]["agent_id"]

                    # ASK: This needs to be if agent not done?
                    if agent_id in agents_need_action:
                        agent_obs = decision_steps[agent_id].obs
                        next_state = (
                            get_grid_based_perception(agent_obs).detach().clone()
                        )
                    else:
                        next_state = None
                        print(f"Got a None next state. team: {team}, episode {episode}")

                    reward = decision_steps[agent_id].reward

                    terminated_agent_ids = terminal_steps.agent_id
                    done = (
                        terminal_steps[agent_id].interrupted
                        if agent_id in terminated_agent_ids
                        else False
                    )

                    action = np.argmax(move_action)
                    # Make a clone of the state tensor. This was overwritten later
                    # making the training loop not work.
                    state = (
                        state_batch_tensor[team_idx, ...].unsqueeze(0).detach().clone()
                    )

                    optimized = agent.step(
                        state, action, reward, next_state, done, episode
                    )
                    # state = (
                    #     next_state  # TODO: This is dobbel up on the for loop futher up
                    # )
                    state_batch_tensor[team_idx, ...] = next_state

                    training_agents[team]["episode_score"] += reward

                    episode_done = done

            scores_all_training_agents = [
                team_info["episode_score"] for team_info in training_agents.values()
            ]

            scores_window.append(scores_all_training_agents)  # save most recent score
            scores.append(scores_all_training_agents)  # save most recent score

            # CRITICAL: This line has an error if no learning has been done. Not enough samples in memory.
            losses.append(agent.cumulative_loss.item())  # save most recent loss

            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            if verbose:
                avg_score = np.mean(scores_window)

                print(f"Episode stats: ")
                print(
                    f"Average Score last {len(scores_window)} episodes: {avg_score:.2f}"
                )
                print(f"Last loss: {agent.cumulative_loss}")
                print(f"Number of transistions in memory: {len(agent.memory)}")

                # min_vals, max_vals = min_max_conf_from_dataset(min_max_conf)
                # print_min_max_conf(min_vals, max_vals)

            if len(losses) > 1:
                plot_loss_from_list(
                    losses,
                    labels=["train"],
                    env_name=config["env_name"],
                    result_dir=results_directory,
                )

            if len(scores) > 1:
                plot_scores_from_nested_list(
                    scores,
                    labels=["train"],
                    env_name=config["env_name"],
                    result_dir=results_directory,
                )

            if early_stop is None:
                continue

            avg_score = np.mean(scores_window)
            if avg_score >= early_stop and episode > 10:
                print(
                    f"\nEnvironment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}"
                )

                break

            if wandb:
                wandb.log({"average_score": avg_score})

    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        print("-" * 100)
        print("Exception has occured !!")
        print("Learning was interrupted. Please wait while the model is saved.")
        print("-" * 100)
        # TODO: Add save model weights, logs, checkoint etc
    finally:
        if wandb:
            wandb.finish()

        env.close()
        # CRITICAL: Save model here and the nessasary values
        save_model(agent.policy_net, agent.model_param["models_dir"])
        # save_dict_to_json(agent.model_param, f"{parameter_directory}/model_param.json")
        # save_dict_to_json(agent.config, f"{parameter_directory}/config.json")
        # save_dict_to_json(agent.dqn_param, f"{parameter_directory}/dqn_param.json")
        # save_dict_to_json(
        #     epsilon_greedy_param, f"{parameter_directory}/epsilon_greedy_param.json"
        # )

        print("Model is saved, parameters is saved & the Environment is closed...")

    return scores, episode, scores_window, losses


def visualize_trained_model(env, agent, config, verbose):
    num_visual_episodes = config["visualize"]["episodes"]

    team_name_list = list(env.behavior_specs.keys())
    num_teams = len(team_name_list)
    decision_steps, _ = env.get_steps(team_name_list[-1])

    num_agents_on_teams = len(decision_steps.agent_id)
    num_total_agents = num_teams * num_agents_on_teams

    state_size = agent.model_param["input_size"]
    state_batch_tensor = torch.zeros((num_total_agents, *state_size))

    try:
        if verbose:
            print(
                f"[INFO] Number of parallell environments during visualization: {num_teams}"
            )

        for episode in range(1, num_visual_episodes + 1):
            if verbose:
                print(f"\nEpisode {episode}/{num_visual_episodes} started")

            env.reset()  # TODO: Test with and without this

            for _, team in enumerate(team_name_list):
                decision_steps, _ = env.get_steps(team)
                agents_need_action = decision_steps.agent_id
                for agent_id in agents_need_action:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs)
                    state_batch_tensor[agent_id, ...] = state

            episode_done = False
            while not episode_done:
                act = agent.act(state_batch_tensor, num_agents=num_total_agents)
                move_action, laser_action = act

                for _, team in enumerate(team_name_list):
                    decision_steps, _ = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    for agent_id in agents_need_action:
                        agent_move_action = move_action[agent_id, ...]
                        agent_laser_action = laser_action[agent_id, ...]
                        env.set_action_for_agent(
                            team,
                            agent_id,
                            ActionTuple(agent_move_action, agent_laser_action),
                        )

                env.step()

                for _, team in enumerate(team_name_list):
                    decision_steps, terminal_steps = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    for agent_id in agents_need_action:
                        agent_obs = decision_steps[agent_id].obs
                        next_state = get_grid_based_perception(agent_obs)
                        state = next_state
                        state_batch_tensor[agent_id, ...] = state

                    terminated_agent_ids = terminal_steps.agent_id
                    done = True if len(terminated_agent_ids) > 0 else False
                    episode_done = done

    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        print(ex)
        print("-" * 100)
        print("Exception has occured !!")
        print("Visualizing was interrupted.")
        print("-" * 100)
        env.close()


if __name__ == "__main__":
    main()
