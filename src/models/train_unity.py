import os
import time
import json
import wandb
import torch
import random

import numpy as np

from sys import platform
from pathlib import Path
from collections import deque

# Local import
from utils.agent import Agent
from utils.ppo_agent import PPO_Agent
from utils.save_utils import save_model, save_dict_to_json, save_list_to_json

from small_dqn import small_DQN
from ee_cnn_residual import EE_CNN_Residual
from resnet_dqn import ResNet_DQN
from resnet_original import ResNet

from utils.data_utils import (
    min_max_conf_from_dataset,
    get_grid_based_perception,
    get_grid_based_perception_numpy,
)
from utils.print_utils import print_min_max_conf, print_cost_of_exits, get_time_hh_mm_ss
from visualization.visualize import (
    plot_scores_from_list,
    plot_loss_from_list,
    plot_grid_based_perception,
    plot_scores_from_nested_list,
)

from visualization.visualize_unity import visualize_trained_model
from evaluate_unity import evaluate_trained_model

from src.models.utils.flops_counter import get_model_complexity_info

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


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def get_avalible_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    DEVICE = get_avalible_device()

    print(f"[INFO] Device is: {DEVICE}")

    model_param = {
        "model_class_name": "EE_CNN_Residual",  # EE_CNN_Residual or small_DQN or ResNet_DQN or ResNet
        "loss_function": "v4",
        "num_ee": 0,
        "repetitions": [2, 2, 2, 2],
        "init_planes": 64,
        "planes": [64, 128, 256, 512],
        "distribution": "pareto",
        # "numbOfCPUThreadsUsed": 10,  # Number of cpu threads use in the dataloader
        "models_dir": None,
        "mode_setups": {"train": True, "eval": True, "visualize": False},
        "manual_seed": 350,  # TODO: Seed everything
        "device": DEVICE,
    }

    model_type = globals()[model_param["model_class_name"]]

    config = {
        "env_name": "FoodCollector",
        "ppo": False,
        "use_build": True,
        "no_graphics": True,
        "laser_length": 1,
        "agent_scale": 1,
        "prioritized_memory": False,
        "memory_size": 25_000,  # 25_000,  # 10_000
        "minimal_memory_size": 512,  # Either batch_size or minimal_memory_size before training
        "batch_size": 128,  # Training batch size
        "num_episodes": 500,
        "benchmarks_mean_reward": None,
        "optimizer": "adam",  # 'SGD' | 'adam' | 'RMSprop' | 'adamW'
        "learning_rate": {
            "lr": 1e-4,  # TUNE: 0.0001 original
            "lr_critic": 0.0001,
        },  # learning rate to the optimizer
        "weight_decay": 0.00001,  # weight_decay value # TUNE: originally 0.00001
        "use_lr_scheduler": False,
        "scheduler_milestones": [75, 200],  # 45,70 end at 80? or 60, 80
        "scheduler_factor": 0.1,
        "clip_gradients": True,
        "max_grad_norm": 1,
        "multiple_epochs": True,
        "num_epochs": 3,
        "print_range": 10,
        "visualize": {
            "episodes": 10,
        },
        "eval": {
            "episodes": 5,
            "every-n-th-episode": 50,
            "all_agents_active": False,
        },
    }

    dqn_param = {
        "gamma": 0.999,  # Original: 0.99,
        "tau": 0.001,  # TUNE: 0.005 original,  # TODO: Try one more 0. 0.05 (5e-2) previous
        "update_every": 4,
    }

    epsilon_greedy_param = {
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "warm_start": 4,
    }

    TRAIN_MODEL = model_param["mode_setups"]["train"]
    VISUALIZE_MODEL = model_param["mode_setups"]["visualize"]
    # EVAL_MODEL = model_param["mode_setups"]["eval"]

    TIMESTAMP = None  # int(1694004681)

    VERBOSE = True

    # Unity environment spesific
    # float_parameter_channel = EnvironmentParametersChannel()
    stats_side_channel = StatsSideChannel()

    engine_config_channel = EngineConfigurationChannel()
    engine_config_channel.set_configuration_parameters(time_scale=10)

    SIDE_CHANNELS = [
        engine_config_channel,
        stats_side_channel,
    ]  # , float_parameter_channel

    if config["use_build"]:
        if platform == "linux" or platform == "linux2":
            relative_path = (
                "builds/Linus_FoodCollector_1_env_no_respawn_headless.x86_64"
            )
            # relative_path = (
            #     "builds/Linus_FoodCollector_4_envs_no_respawn_headless.x86_64"
            # )
            FILE_NAME = relative_path

        else:
            relative_path = "builds/FoodCollector_1_env_no_respawn.app"
            # relative_path = "builds/FoodCollector_4_no_respawn.app"
            # relative_path = "builds/FoodCollector_1_env_no_respawn_overhead.app"
            FILE_NAME = relative_path

    else:
        FILE_NAME = None

    random_worker_id = random.randint(0, 1250)
    print(f"[INFO] Random worker id: {random_worker_id}")

    # set_seed(model_param["manual_seed"])

    env = UnityEnvironment(
        file_name=FILE_NAME,
        side_channels=SIDE_CHANNELS,
        # seed=model_param["manual_seed"],
        no_graphics=config["no_graphics"],
        worker_id=random_worker_id,
    )

    # Unity environment spesific
    # float_parameter_channel.set_float_parameter("laser_length", config["laser_length"])
    # float_parameter_channel.set_float_parameter("agent_scale", config["agent_scale"])

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

    if DEVICE != "mps":
        run_wandb = wandb.init(
            project="Master-thesis",
            config={**model_param, **config, **dqn_param, **epsilon_greedy_param},
        )
    else:
        run_wandb = None

    if TRAIN_MODEL:
        timestamp = int(time.time())
        print(f"[INFO] Results added to folder: {timestamp}")

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

        use_ppo = config["ppo"]

        if use_ppo:
            print(f"[INFO] Initalizing PPO policy network of type {model_type}")
            ppo_ee_policy_net = model_type(
                # frames_history=2,
                num_ee=model_param["num_ee"],
                init_planes=model_param["init_planes"],
                planes=model_param["planes"],
                input_shape=model_param["input_size"],
                num_classes=model_param["num_classes"],
                repetitions=model_param["repetitions"],
                distribution=model_param["distribution"],
            ).to(DEVICE)

            old_ppo_ee_policy_net = model_type(
                # frames_history=2,
                num_ee=model_param["num_ee"],
                init_planes=model_param["init_planes"],
                planes=model_param["planes"],
                input_shape=model_param["input_size"],
                num_classes=model_param["num_classes"],
                repetitions=model_param["repetitions"],
                distribution=model_param["distribution"],
            ).to(DEVICE)

            if VERBOSE:
                print("[INFO] Cost of the initalized model")
                print_cost_of_exits(ppo_ee_policy_net)

            # ASK: This is important to get the networks initalized with the same weigths
            print("[INFO] Copying weight to get same weights in both policy nets")
            old_ppo_ee_policy_net.load_state_dict(ppo_ee_policy_net.state_dict())

            print("[INFO] Initalizing critic network")
            critic_net = torch.nn.Sequential(
                torch.nn.Linear(model_param["input_size"], 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            ).to(DEVICE)

            agent = PPO_Agent(
                ppo_ee_policy_net,
                old_ppo_ee_policy_net,
                critic_net,
                model_param=model_param,
                config=config,
            )

        else:
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

            if VERBOSE and ee_policy_net.complexity:
                print("[INFO] Cost of the initalized model")
                print_cost_of_exits(ee_policy_net)

            ee_policy_net.train()
            ee_target_net.train()

            # ASK: This is important to get the networks initalized with the same weigths
            # print("[INFO] Copying weight from target net to policy net")
            # ee_target_net.load_state_dict(ee_policy_net.state_dict())

            print("[INFO] Initalizing a Agent object")
            agent = Agent(
                ee_policy_net,
                ee_target_net,
                model_param=model_param,
                config=config,
                dqn_param=dqn_param,
            )

            startTime = time.time()
            print(f"[INFO] Started training @ {time.ctime(startTime)}")

            save_dict_to_json(model_param, f"./{parameter_directory}/model_param.json")
            save_dict_to_json(config, f"./{parameter_directory}/config.json")
            save_dict_to_json(dqn_param, f"./{parameter_directory}/dqn_param.json")
            save_dict_to_json(
                epsilon_greedy_param,
                f"./{parameter_directory}/epsilon_greedy_param.json",
            )

            save_model(
                agent.policy_net,
                agent.model_param["models_dir"],
                model_type="untrained",
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

        engine_config_channel.set_configuration_parameters(time_scale=1)

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

        model_type = globals()[model_param["model_class_name"]]

        print(f"[INFO] Loading the trained policy net of type {model_type}")

        ee_policy_net = model_type(
            num_ee=model_param["num_ee"],
            init_planes=model_param["init_planes"],
            planes=model_param["planes"],
            input_shape=model_param["input_size"],
            num_classes=model_param["num_classes"],
            repetitions=model_param["repetitions"],
            distribution=model_param["distribution"],
        )

        models_directory = model_param["models_dir"]
        model_file = f"{models_directory}/last_model.pt"
        # model_file = f"{models_directory}/untrained_model.pt"
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

        visualize_trained_model(env, agent, config, verbose=VERBOSE)


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
    eps_end = epsilon_greedy_param["eps_end"]
    eps_start = epsilon_greedy_param["eps_start"]
    eps_decay = epsilon_greedy_param["eps_decay"]
    warm_start = epsilon_greedy_param["warm_start"]
    eps = eps_start

    print_range = config["print_range"]
    early_stop = config["benchmarks_mean_reward"]

    num_episodes = config["num_episodes"]

    scores, losses = list(), list()  # list containing scores/losses from each episode
    scores_window = deque(maxlen=print_range)

    # Evaluate variables
    evaluate_model = agent.model_param["mode_setups"]["eval"]
    evaluate_every_n_th_episode = config["eval"]["every-n-th-episode"]

    try:
        team_name_list = list(env.behavior_specs.keys())
        num_teams = len(team_name_list)

        if verbose:
            print(
                f"[INFO] Number of parallell environments during training: {num_teams}"
            )

        state_size = agent.model_param["input_size"]
        state_batch_tensor = torch.zeros((num_teams, *state_size))

        for episode in range(1, num_episodes + 1):
            training_agents = dict()
            conf_min_max = list()
            random_actions = 0

            if verbose:
                print(f"\nEpisode {episode}/{num_episodes} started")

            env.reset()

            for team_idx, team in enumerate(team_name_list):
                decision_steps, _ = env.get_steps(team)
                training_agents[team] = {
                    "agent_id": decision_steps.agent_id[
                        -1
                    ],  # random agent from each team
                    "episode_score": 0,
                    "exit_points": [0] * (agent.policy_net.num_ee + 1),
                }
                agent_id = training_agents[team]["agent_id"]
                agent_obs = decision_steps[agent_id].obs
                state = get_grid_based_perception(agent_obs).detach().clone()
                state_batch_tensor[team_idx, ...] = state

            # min_max_conf = list()
            episode_done = False
            while not episode_done:
                move_action, laser_action, confs, exits, costs = agent.act(
                    state_batch_tensor.detach().clone(),
                    epsilon=eps,
                    num_agents=num_teams,
                )

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
                        (state_batch_tensor[team_idx, ...])
                        .unsqueeze(0)
                        .detach()
                        .clone()
                    )

                    if isinstance(agent, PPO_Agent):
                        log_prob = 0
                        agent.step(state, action, log_prob, reward, done)
                    else:
                        optimized = agent.step(state, action, reward, next_state, done)
                        if optimized:
                            conf_min_max.append(agent.train_conf)

                    state_batch_tensor[team_idx, ...] = next_state.detach().clone()

                    training_agents[team]["episode_score"] += reward

                    if isinstance(exits, int):
                        training_agents[team]["exit_points"][exits] += 1
                    elif isinstance(exits, torch.Tensor):
                        exit = exits[team_idx]
                        training_agents[team]["exit_points"][exit] += 1
                    elif exits is None:
                        random_actions += 1
                    else:
                        print("The type of exits are not supported at this point")

                    episode_done = done

            scores_all_training_agents = [
                team_info["episode_score"] for team_info in training_agents.values()
            ]

            scores_window.append(scores_all_training_agents)  # save most recent score
            scores.append(scores_all_training_agents)  # save most recent score

            # CRITICAL: This line has an error if no learning has been done. Not enough samples in memory.
            losses.append(agent.cumulative_loss.item())  # save most recent loss

            if warm_start is not None and episode > warm_start:
                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            if warm_start is None:
                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            avg_score = np.mean(scores_window)

            if wandb:
                wandb.log(
                    {
                        "loss": losses[-1],
                        "average_score": avg_score,
                        "min_last_score": min(scores_all_training_agents),
                        "max_last_score": max(scores_all_training_agents),
                        "epsilon": eps,
                        "Mean Q targets": torch.mean(torch.abs(agent.last_Q_targets)),
                        "Mean Q expected": torch.mean(torch.abs(agent.last_Q_expected)),
                    }
                )

            if verbose:
                print(f"Episode stats: ")
                print(
                    f"Average Score last {len(scores_window)} episodes: {avg_score:.2f}"
                )
                print(f"Last loss: {agent.cumulative_loss}")

                min_vals, max_vals = min_max_conf_from_dataset(conf_min_max)
                print_min_max_conf(min_vals, max_vals)

                print(f"Number of transistions in memory: {len(agent.memory)}")

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

            if verbose:
                print(
                    f"Last Q target values: {torch.mean(torch.abs(agent.last_Q_targets))}"
                )
                print(
                    f"Last Q expected values: {torch.mean(torch.abs(agent.last_Q_expected))}"
                )

            if early_stop:
                if avg_score >= early_stop and episode > 10:
                    print(
                        f"\nEnvironment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}"
                    )

                    break

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

        save_list_to_json(scores, f"./{results_directory}/scores.json")
        save_list_to_json(losses, f"./{results_directory}/losses.json")

        print("Model is saved, parameters is saved & the Environment is closed...")

    return scores, episode, scores_window, losses


if __name__ == "__main__":
    main()
