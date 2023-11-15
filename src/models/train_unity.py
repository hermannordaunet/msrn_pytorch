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
from exploding_ee_cnn import Exploding_EE_CNN_Residual
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
from evaluate_unity import (
    evaluate_trained_model,
    extract_exit_points_from_agents,
    extract_scores_for_all_agents,
    extract_one_agent_each_team,
)

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
        "model_class_name": "EE_CNN_Residual",  # EE_CNN_Residual, small_DQN, ResNet_DQN, ResNet, Exploding_EE_CNN_Residual
        "loss_function": "v5", #"v3" "v5"
        "exit_loss_function": "loss_exit", #"loss_exit", None
        "num_ee": 5,
        "exit_threshold": [0.7],
        "repetitions": [3, 4, 6, 3],  # [2, 2, 2, 2] resnet18, [3, 4, 6, 3] resnet34
        "init_planes": 64,
        "planes": [64, 128, 256, 512],
        "distribution": "linear",
        "models_dir": None,
        "mode_setups": {"train": True, "eval": True, "visualize": False},
        "manual_seed": 350,  # TODO: Seed ezverything
        "device": DEVICE,
    }

    model_type = globals()[model_param["model_class_name"]]

    config = {
        "env_name": "FoodCollector",
        "double_dqn": True,
        "ppo": False,
        "use_build": True,
        "no_graphics": True,
        "laser_length": 1,
        "agent_scale": 1,
        "prioritized_memory": False,
        "memory_size": 35_000,  # 25_000,  # 10_000
        "minimal_memory_size": 999,  # Either batch_size or minimal_memory_size before training
        "batch_size": 256,  # Training batch size
        "benchmarks_mean_reward": None,
        "optimizer": "adam",  # 'SGD' | 'adam' | 'RMSprop' | 'adamW'
        "learning_rate": {
            "lr": 0.0001,  # TUNE: 0.0001 original
            "lr_critic": 0.0001,
            "lr_exit": 0.001,
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
        "train": {
            "episodes": 1000,
            "all_agents_active": True,
        },
        "eval": {
            "episodes": 10,
            "every-n-th-episode": 30,
            "all_agents_active": True,
            "one_of_each_exit": False,
            "random_agent": False,
        },
        "visualize": {
            "episodes": 10,
            "all_agents_active": True,
        },
    }

    dqn_param = {
        "gamma": 0.999,  # Original: 0.99,
        "tau": 0.001,  # TUNE: 0.005 original,  # TODO: Try one more 0. 0.05 (5e-2) previous
        "update_every": 25,
    }

    epsilon_greedy_param = {
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "warm_start": 4,
    }

    TRAIN_MODEL = model_param["mode_setups"]["train"]
    EVAL_MODEL = model_param["mode_setups"]["eval"]
    VISUALIZE_MODEL = model_param["mode_setups"]["visualize"]

    # TIMESTAMP = int(1699472908)

    VERBOSE = True

    # Unity environment spesific
    float_parameter_channel = EnvironmentParametersChannel()
    stats_side_channel = StatsSideChannel()

    engine_config_channel = EngineConfigurationChannel()
    engine_config_channel.set_configuration_parameters(time_scale=10)

    SIDE_CHANNELS = [
        engine_config_channel,
        stats_side_channel,
        float_parameter_channel,
    ]

    if config["use_build"]:
        if platform == "linux" or platform == "linux2":
            # relative_path = "builds/Linus_FoodCollector_1_env_no_respawn_headless.x86_64"
            # relative_path = "builds/Linus_FoodCollector_1_envs_no_respawn_wall_penalty_2_and_-4_reward_7_agents.x86_64"
            # relative_path = "builds/Linus_FoodCollector_1_envs_no_respawn_wall_penalty_2_and_-4_reward_6_agents.x86_64"
            # relative_path = "builds/Linus_FoodCollector_1_envs_no_respawn_wall_penalty_2_and_-4_no_wall-hit_reward_6_agents.x86_64"
            # relative_path = "builds/Linus_FoodCollector_1_envs_no_respawn_wall_penalty_2_and_-4_no_wall-hit_reward_8_agents.x86_64"
            # relative_path = "builds/Linus_FoodCollector_1_envs_no_respawn_wall_penalty_2_and_-4_no_wall-hit_reward_7_agents.x86_64"
            relative_path = "builds/Linus_FoodCollector_1_env_no_respawn_wall_penalty_2_and_-4_reward.x86_64"
        else:
            # relative_path = "builds/FoodCollector_1_env_no_respawn.app"
            # relative_path = "builds/FoodCollector_4_no_respawn.app"
            # relative_path = "builds/FoodCollector_1_env_no_respawn_overhead.app"
            # relative_path = "builds/FoodCollector_1_env_respawn_wall_penalty_2_and_-4_reward_7_agents.app"
            relative_path = "builds/FoodCollector_1_env_respawn_wall_penalty_2_and_-4_reward_6_agents.app"
            # relative_path = "builds/FoodCollector_1_env_respawn_no_wall_penalty_2_and_-4_reward_8_agents.app"
    else:
        relative_path = None

    FILE_NAME = relative_path

    random_worker_id = random.randint(0, 1250)
    print(f"[INFO] Random worker id: {random_worker_id}")

    # set_seed(model_param["manual_seed"])

    if DEVICE != "mps" and TRAIN_MODEL:
        run_wandb = wandb.init(
            project="Master-thesis",
            config={**model_param, **config, **dqn_param, **epsilon_greedy_param},
        )
    else:
        run_wandb = None

    if TRAIN_MODEL:
        env = UnityEnvironment(
            file_name=FILE_NAME,
            side_channels=SIDE_CHANNELS,
            # seed=model_param["manual_seed"],
            no_graphics=config["no_graphics"],
            worker_id=random_worker_id,
        )

        # Unity environment spesific
        float_parameter_channel.set_float_parameter(
            "laser_length", config["laser_length"]
        )
        float_parameter_channel.set_float_parameter(
            "agent_scale", config["agent_scale"]
        )

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

        c, w, h = observation_spec[-1].shape[::-1]
        model_param["input_size"] = (c, w, h)
        model_param["num_classes"] = continuous_size
        # channels, screen_width, screen_height = input_size

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
                exit_threshold=model_param["exit_threshold"],
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
                exit_threshold=model_param["exit_threshold"],
                initalize_parameters=False,
            ).to(DEVICE)

            if VERBOSE and ee_policy_net.complexity:
                print("[INFO] Cost of the initalized model")
                print_cost_of_exits(ee_policy_net)

            ee_policy_net.train()
            ee_target_net.train()

            # ASK: This is important to get the networks initalized with the same weigths
            print("[INFO] Copying weight from target net to policy net")
            ee_target_net.load_state_dict(ee_policy_net.state_dict())

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

    if EVAL_MODEL and not TRAIN_MODEL:
        try:
            env.close()
        except:
            print("Environment already closed")

        engine_config_channel.set_configuration_parameters(time_scale=10)

        env = UnityEnvironment(
            file_name=FILE_NAME,
            side_channels=SIDE_CHANNELS,
            # seed=model_param["manual_seed"],
            no_graphics=config["no_graphics"],
            worker_id=random_worker_id,

        )

        # Unity environment spesific
        float_parameter_channel.set_float_parameter(
            "laser_length", config["laser_length"]
        )
        float_parameter_channel.set_float_parameter(
            "agent_scale", config["agent_scale"]
        )

        env.reset()

        timestamp = int(time.time())
        print(f"[INFO] Eval results added to folder: {timestamp}")

        eval_results_directory = Path(f"./evaluation_results/{timestamp}/")

        # Check if the directory exists
        if not eval_results_directory.exists():
            # If it doesn't exist, create it
            eval_results_directory.mkdir(parents=True)

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
        # config = load_json_as_dict(f"{parameter_directory}/config.json")
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
            exit_threshold=model_param["exit_threshold"],
        )

        models_directory = model_param["models_dir"]
        model_file = f"{models_directory}/last_model.pt"
        # model_file = f"{models_directory}/untrained_model.pt"
        print(f"[INFO] Loading weights from {model_file}")

        eval_threshold = model_param["exit_threshold"]
        print(f"[INFO] Evaluating with exit thresholds: {eval_threshold}")

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

        agent = Agent(
            ee_policy_net,
            ee_target_net,
            model_param=model_param,
            config=config,
            dqn_param=dqn_param,
        )

        evalStartTime = time.time()
        print(f"[INFO] Started training @ {time.ctime(evalStartTime)}")

        evaluate_trained_model(
            env,
            agent,
            config,
            results_directory=eval_results_directory,
            current_episode=None,
            verbose=VERBOSE,
        )

        evalEndTime = time.time()
        total_sec = evalEndTime - evalStartTime
        print(
            f"[INFO] total time taken to eval the model: {get_time_hh_mm_ss(total_sec)} sec"
        )

        try:
            env.close()
        except:
            print("Environment already closed")

    if VISUALIZE_MODEL:
        # Close old env and start fresh
        try:
            env.close()
        except:
            print("Environment already closed")

        engine_config_channel.set_configuration_parameters(time_scale=1)

        env = UnityEnvironment(
            file_name=FILE_NAME,
            side_channels=SIDE_CHANNELS,
            # seed=model_param["manual_seed"],
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
            exit_threshold=model_param["exit_threshold"],
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

        try:
            env.close()
        except:
            print("Environment already closed")


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

    (
        scores,
        losses,
        exit_losses,
        moving_avg_score,
        mean_q_targets,
        mean_q_expected,
        max_train_exit_confs,
        min_train_exit_confs,
        mean_train_exit_confs,
    ) = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )  # list containing scores/losses from each episode
    scores_window = deque(maxlen=print_range)

    best_model_score = float("-inf")

    # Evaluate variables
    evaluate_model = agent.model_param["mode_setups"]["eval"]
    evaluate_every_n_th_episode = config["eval"]["every-n-th-episode"]

    num_train_episodes = config["train"]["episodes"]
    all_train_agents_active = config["train"]["all_agents_active"]

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
                f"[INFO] Number of parallell environments during training: {num_teams}"
            )

        for episode in range(1, num_train_episodes + 1):
            training_agents = dict()
            conf_min_max = list()

            if verbose:
                print(f"\nEpisode {episode}/{num_train_episodes} started")

            env.reset()

            for team_idx, team in enumerate(team_name_list):
                training_agents[team] = dict()

                decision_steps, _ = env.get_steps(team)
                agent_id = decision_steps.agent_id[-1]  # random agent from each team

                agents_need_action = decision_steps.agent_id
                for agent_id in agents_need_action:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs).detach().clone()
                    state_batch_tensor[agent_id, ...] = state

                    training_agents[team][agent_id] = {
                        "bad_food": 0,
                        "good_food": 0,
                        "wall_hit": 0,
                        "episode_score": 0,
                        "random_actions": 0,
                        "exit_points": [0] * (agent.policy_net.num_ee + 1),
                        "move_actions": np.zeros((1, 1, 3)),
                    }

            # min_max_conf = list()
            episode_done = False

            if not all_train_agents_active:
                active_agent_id = extract_one_agent_each_team(training_agents)

            while not episode_done:
                # move_action, laser_action, confs, exits, costs
                move_actions, laser_actions, confs, exits, costs = agent.act(
                    state_batch_tensor.detach().clone(),
                    epsilon=eps,
                    num_agents=num_total_agents,
                    eval_agent=False,
                )

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, _ = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    if all_train_agents_active:
                        agents_to_act = agents_need_action
                        agent_ids_to_print = None
                    else:
                        agents_to_act = [active_agent_id[team_idx]]
                        agent_ids_to_print = agents_to_act

                    for agent_id in agents_to_act:
                        agent_move_action = move_actions[agent_id, ...]
                        agent_laser_action = laser_actions[agent_id, ...]

                        training_agents[team][agent_id]["move_actions"] += agent_move_action

                        env.set_action_for_agent(
                            team,
                            agent_id,
                            ActionTuple(agent_move_action, agent_laser_action),
                        )

                env.step()

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, terminal_steps = env.get_steps(team)
                    agents_need_action_after_step = decision_steps.agent_id
                    if all_train_agents_active:
                        agents_to_act = agents_need_action_after_step
                        agent_ids_to_print = None
                    else:
                        agents_to_act = [active_agent_id[team_idx]]
                        agent_ids_to_print = agents_to_act

                    for agent_id in agents_need_action_after_step:
                        agent_dict = training_agents[team][agent_id]
                        agent_obs = decision_steps[agent_id].obs
                        if agent_id not in agents_need_action:
                            next_state = None
                            print(
                                f"Got a None next state. team: {team}, episode {episode}"
                            )
                        else:
                            next_state = (
                                get_grid_based_perception(agent_obs).detach().clone()
                            )

                        reward = decision_steps[agent_id].reward
                        action = np.argmax(move_actions[agent_id, ...])

                        state = (
                            (state_batch_tensor[agent_id, ...])
                            .unsqueeze(0)
                            .detach()
                            .clone()
                        )

                        terminated_agent_ids = terminal_steps.agent_id
                        done = (
                            terminal_steps[agent_id].interrupted
                            if agent_id in terminated_agent_ids
                            else False
                        )

                        if isinstance(agent, PPO_Agent):
                            log_prob = 0
                            agent.step(state, action, log_prob, reward, done)
                        else:
                            optimized = agent.step(
                                state, action, reward, next_state, done
                            )
                            if optimized:
                                conf_min_max.append(agent.train_conf)

                        state_batch_tensor[agent_id, ...] = next_state.detach().clone()

                        if reward == -1.0:
                            agent_dict["wall_hit"] += 1

                        if reward == -4.0:
                            agent_dict["bad_food"] += 1

                        if reward > 0.0:
                            agent_dict["good_food"] += 1

                        if reward < -4.0:
                            agent_dict["wall_hit"] += 1
                            agent_dict["bad_food"] += 1

                        if float(reward) != 0.0:
                            agent_dict["episode_score"] += reward

                        if isinstance(exits, int):
                            agent_dict["exit_points"][exits] += 1
                        elif isinstance(exits, torch.Tensor):
                            exit = exits[team_idx]
                            agent_dict["exit_points"][exit] += 1
                        elif exits is None:
                            agent_dict["random_actions"] += 1
                        else:
                            print("The type of exits are not supported at this point")

                        episode_done = done

            (
                scores_all_training_agents,
                bad_food,
                good_food,
            ) = extract_scores_for_all_agents(
                training_agents,
                active_agents=agent_ids_to_print,
                food_info=True,
                flatten=True,
            )

            scores_window.append(scores_all_training_agents)  # save most recent score
            scores.append(scores_all_training_agents)  # save most recent score

            mean_q_targets.append(torch.mean(torch.abs(agent.last_Q_targets)))
            mean_q_expected.append(torch.mean(torch.abs(agent.last_Q_expected)))

            # CRITICAL: This line has an error if no learning has been done. Not enough samples in memory.
            losses.append(agent.full_net_loss.item())  # save most recent loss
            
            exit_loss = agent.cumulative_exits_loss
            if exit_loss is not None:
                exit_losses.append(exit_loss.item())

            if warm_start is not None and episode > warm_start:
                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            if warm_start is None:
                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            avg_score = np.mean(scores_window)
            moving_avg_score.append(avg_score)

            if avg_score > best_model_score:
                save_model(
                    agent.policy_net,
                    agent.model_param["models_dir"],
                    model_type="best",
                )
                best_model_score = avg_score
                best_model_episode = episode

            if wandb:
                wandb.log(
                    {
                        "loss": losses[-1],
                        # "exit_loss": exit_losses[-1],
                        "average_score": moving_avg_score[-1],
                        "min_last_score": min(scores_all_training_agents),
                        "max_last_score": max(scores_all_training_agents),
                        "good_food": np.mean(good_food),
                        "bad_food": np.mean(bad_food),
                        "epsilon": eps,
                        "Mean Q targets": mean_q_targets[-1],
                        "Mean Q expected": mean_q_expected[-1],
                    }
                )

            if verbose:
                print(f"Episode stats: ")
                print(
                    f"Average Score last {len(scores_window)} episodes: {moving_avg_score[-1]:.2f}"
                )
                print(f"Last loss: {agent.full_net_loss}")
                print(f"Cumulative exit loss: {agent.cumulative_exits_loss}")

                min_vals, max_vals, mean_vals = min_max_conf_from_dataset(
                    conf_min_max, include_last=False
                )
                print_min_max_conf(min_vals, max_vals, mean_vals)

                max_train_exit_confs.append(max_vals)
                min_train_exit_confs.append(min_vals)
                mean_train_exit_confs.append(mean_vals)

                extract_exit_points_from_agents(
                    training_agents,
                    active_agent_id=agent_ids_to_print,
                    include_reward=True,
                    include_food_info=True,
                    include_wall_info=True,
                    mode="TRAIN",
                    print_out=True,
                    random_actions=True,
                )

                print(f"Number of transistions in memory: {len(agent.memory)}")

            if len(losses) > 1:
                plot_loss_from_list(
                    losses,
                    labels=["train"],
                    env_name=config["env_name"],
                    result_dir=results_directory,
                )

            if len(exit_losses) > 1:
                plot_loss_from_list(
                    exit_losses,
                    labels=["train"],
                    env_name=config["env_name"],
                    result_dir=results_directory,
                    loss_type="Cumulative_Exit",
                )

            if len(scores) > 1:
                plot_scores_from_nested_list(
                    scores,
                    labels=["train"],
                    env_name=config["env_name"],
                    result_dir=results_directory,
                )

            if verbose:
                print(f"Last Q target values: {mean_q_targets[-1]}")
                print(f"Last Q expected values: {mean_q_expected[-1]}")

            if early_stop:
                if moving_avg_score[-1] >= early_stop and episode > 10:
                    print(
                        f"\nEnvironment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}"
                    )

                    break

            evaluate_this_episode = not episode % evaluate_every_n_th_episode
            done_training = episode == num_train_episodes
            run_evaluation_now = evaluate_this_episode or done_training

            if evaluate_model and run_evaluation_now:
                message = ""

                if done_training:
                    message += "\nEvaluation after last episode"

                message += "\nEvaluation started"

                if verbose:
                    print(message)

                evaluate_trained_model(
                    env, agent, config, current_episode=episode, verbose=verbose
                )

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

        try:
            env.close()
        except:
            print("Environment already closed")

        # CRITICAL: Save model here and the nessasary values
        save_model(agent.policy_net, agent.model_param["models_dir"])

        save_list_to_json(scores, f"./{results_directory}/scores.json")
        save_list_to_json(losses, f"./{results_directory}/losses.json")
        save_list_to_json(exit_losses, f"./{results_directory}/exit_losses.json")
        save_list_to_json([episode], f"./{results_directory}/episode.json")
        save_list_to_json(scores_window, f"./{results_directory}/scores_window.json")
        save_list_to_json(
            moving_avg_score, f"./{results_directory}/moving_avg_score.json"
        )

        save_list_to_json(
            max_train_exit_confs, f"./{results_directory}/max_train_exit_confs.json"
        )

        save_list_to_json(
            min_train_exit_confs, f"./{results_directory}/min_train_exit_confs.json"
        )

        save_list_to_json(
            mean_train_exit_confs, f"./{results_directory}/mean_train_exit_confs.json"
        )

        print("Model is saved, parameters is saved & the Environment is closed...")
        if best_model_episode:
            print(f"Best model was saved from episode: {best_model_episode}")

    return scores, episode, scores_window, losses


if __name__ == "__main__":
    main()
