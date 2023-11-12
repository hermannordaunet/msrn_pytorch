import torch
import random

import numpy as np

# unity imports
from mlagents_envs.base_env import ActionTuple

from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)

# Local imports
from src.models.utils.data_utils import get_grid_based_perception
from src.models.utils.save_utils import save_list_to_json


def extract_scores_for_all_agents(
    eval_agents: dict, active_agents=None, food_info=False, flatten=False
):
    # Extract the team keys from the eval_agents
    team_keys = list(eval_agents.keys())

    # Extract all episode keys from all teams' dictionaries
    if active_agents is None:
        agent_keys = list()
        for team_key in team_keys:
            agents_in_team = list(eval_agents[team_key].keys())

            agent_keys.append(tuple(agents_in_team))
    else:
        agent_keys = [[item] for item in active_agents]

    if flatten:
        # Flatten the scores into a 1D list
        scores = list()

        if food_info:
            bad_food = list()
            good_food = list()

        for team_id, team_key in enumerate(team_keys):
            for agent_key in agent_keys[team_id]:
                agent_dict = eval_agents[team_key][agent_key]
                scores.append(agent_dict["episode_score"])

                if food_info:
                    bad_food.append(agent_dict["bad_food"])
                    good_food.append(agent_dict["good_food"])
    else:
        # Initialize a 2D list with zeros based on the number of teams and episodes
        num_teams = len(team_keys)
        scores = [None] * num_teams

        if food_info:
            bad_food = [None] * num_teams
            good_food = [None] * num_teams

        # Iterate through teams and episodes to fill the scores
        for i, team_key in enumerate(team_keys):
            scores[i] = [0] * len(agent_keys[i])
            for j, episode_key in enumerate(agent_keys[i]):
                if episode_key in eval_agents[team_key]:
                    agent_dict = eval_agents[team_key][episode_key]
                    scores[i][j] = agent_dict["episode_score"]
                    bad_food[i][j] = agent_dict["bad_food"]
                    good_food[i][j] = agent_dict["good_food"]

    if food_info:
        return scores, bad_food, good_food

    return scores


def extract_one_agent_each_team(eval_agents: dict):
    team_keys = list(eval_agents.keys())
    one_agent_keys = list()
    for team_key in team_keys:
        agents_in_team = list(eval_agents[team_key].keys())
        one_agent_keys.append(agents_in_team[-1])

    return one_agent_keys


def extract_exit_points_from_agents(
    eval_agents: dict,
    active_agent_id: list() = None,
    include_reward=False,
    include_food_info=False,
    include_wall_info=False,
    include_action_distribution=False,
    mode: str = "EVAL",
    print_out=True,
    random_actions=False,
):
    all_agents_exit_points = []
    all_agents_reward = []
    all_agents_good_food = []
    all_agents_bad_food = []
    all_agents_wall_hits = []
    all_agents_action_dist = []

    for team, team_data in eval_agents.items():
        agent_ids = team_data.keys()
        if active_agent_id is None:
            agents_to_print = team_data.keys()
        else:
            agents_to_print = active_agent_id

        team_agents_exit_points = []
        team_agents_reward = []
        team_agents_good_food = []
        team_agents_bad_food = []
        team_agents_wall_hits = []
        team_agents_action_dist = []

        for agent_id in agent_ids:
            if agent_id in agents_to_print:
                agent_dict = eval_agents[team][agent_id]
                team_agents_exit_points.append(agent_dict["exit_points"])
                team_agents_reward.append(agent_dict["episode_score"])
                team_agents_good_food.append(agent_dict["good_food"])
                team_agents_bad_food.append(agent_dict["bad_food"])
                team_agents_wall_hits.append(agent_dict["wall_hit"])
                team_agents_action_dist.append(agent_dict["move_actions"].tolist()[0][0])

                if print_out:
                    message = f"[{mode}] Agent ID: {agent_id}, Exit Points: {team_agents_exit_points[-1]}"

                    if random_actions:
                        random_actions = agent_dict["random_actions"]
                        message += f", Random Actions: {random_actions}"

                    if include_reward:
                        message += f", Reward: {team_agents_reward[-1]}"

                    if include_food_info:
                        message += f", Good Food: {team_agents_good_food[-1]}, Bad Food: {team_agents_bad_food[-1]}"

                    if include_wall_info:
                        message += f", Wall hits: {team_agents_wall_hits[-1]}"

                    if include_action_distribution:
                        message += f", Action dist: {team_agents_action_dist[-1]}"

                    print(message)

        all_agents_exit_points.append(team_agents_exit_points)
        all_agents_reward.append(team_agents_reward)
        all_agents_good_food.append(team_agents_good_food)
        all_agents_bad_food.append(team_agents_bad_food)
        all_agents_wall_hits.append(team_agents_wall_hits)
        all_agents_action_dist.append(team_agents_action_dist)

    if not print_out:
        return (
            all_agents_exit_points,
            all_agents_reward,
            all_agents_good_food,
            all_agents_bad_food,
            all_agents_wall_hits,
            all_agents_action_dist,
        )


def evaluate_trained_model(
    env, agent, config, results_directory=None, current_episode=None, verbose=False
):
    eval_each_exit = config["eval"]["one_of_each_exit"]
    random_agent = config["eval"]["random_agent"]

    cat_dim_action = 0
    cat_dim_tensors = 1

    message = "[INFO] Evaluating with"
    if eval_each_exit:
        message += " one agent per exit"
    if random_agent:
        message += " and with a random exiting agent."

    message += " Also with one hybrid agent"

    if eval_each_exit:
        print(message)

    was_in_training = False
    if agent.policy_net.training:
        was_in_training = True
        agent.policy_net.eval()

    num_eval_episodes = config["eval"]["episodes"]
    all_agents_active = config["eval"]["all_agents_active"]

    team_name_list = list(env.behavior_specs.keys())
    num_teams = len(team_name_list)
    decision_steps, _ = env.get_steps(team_name_list[-1])

    num_agents_on_teams = len(decision_steps.agent_id)
    num_total_agents = num_teams * num_agents_on_teams
    if eval_each_exit:
        if (agent.policy_net.num_ee + 3) != num_agents_on_teams:
            print(
                "Cannot run this evaluation because there are not enough agents for each exit, one random, and one hybrid"
            )
            print(f"{num_agents_on_teams} agents in the env")
            env.close()

    state_size = agent.model_param["input_size"]
    state_batch_tensor = torch.zeros((num_total_agents, *state_size))

    exit_points, rewards, good_food, bad_food, wall_hit, action_dist = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )

    try:
        if verbose:
            print(
                f"[INFO] Number of parallell environments during evaluation: {num_teams}"
            )


        for episode in range(1, num_eval_episodes + 1):
            eval_agents = dict()

            if verbose:
                print(f"Evaluation episode {episode}/{num_eval_episodes} started")

            env.reset()

            for _, team in enumerate(team_name_list):
                eval_agents[team] = dict()

                decision_steps, _ = env.get_steps(team)
                agents_need_action = decision_steps.agent_id
                for agent_id in agents_need_action:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs).detach().clone()
                    state_batch_tensor[agent_id, ...] = state

                    eval_agents[team][agent_id] = {
                        "bad_food": 0,
                        "good_food": 0,
                        "wall_hit": 0,
                        "episode_score": 0,
                        "exit_points": [0] * (agent.policy_net.num_ee + 1),
                        "agent_confs": [[] for _ in range(agent.policy_net.num_ee + 1)],
                        "move_actions": np.zeros((1, 1, 3)),
                    }

            episode_done = False

            if not all_agents_active:
                active_agent_id = extract_one_agent_each_team(eval_agents)

            while not episode_done:
                # TODO: loop through the agent list if there is a list
                if not eval_each_exit:
                    move_action, laser_action, confs, exits, costs = agent.act(
                        state_batch_tensor, num_agents=num_total_agents, eval_agent=True
                    )
                else:
                    exits = []
                    confs = None
                    move_action = None
                    laser_action = None
                    for exit_ids in range(1, agent.policy_net.num_ee + 1):
                        exit_agent_state = state_batch_tensor[exit_ids].unsqueeze(0)
                        (
                            exit_move_action,
                            exit_laser_action,
                            conf,
                            exit,
                            cost,
                        ) = agent.act(
                            exit_agent_state,
                            num_agents=1,
                            eval_all_exit=True,
                            eval_exit_point=exit_ids,
                        )
                        exits.append(exit)
                        if confs is None:
                            confs = conf
                        else:
                            confs = torch.cat((confs, conf), dim=cat_dim_tensors)

                        if move_action is None:
                            move_action = exit_move_action
                            laser_action = exit_laser_action
                        else:
                            move_action = np.concatenate(
                                (move_action, exit_move_action), axis=cat_dim_action
                            )
                            laser_action = np.concatenate(
                                (laser_action, exit_laser_action), axis=cat_dim_action
                            )

                    exit_agent_state = state_batch_tensor[-3].unsqueeze(0)
                    (
                        exit_move_action,
                        exit_laser_action,
                        conf,
                        exit,
                        cost,
                    ) = agent.act(
                        exit_agent_state,
                        num_agents=1,
                    )

                    exits.append(exit)
                    confs = torch.cat((confs, conf), dim=cat_dim_tensors)
                    move_action = np.concatenate(
                        (move_action, exit_move_action), axis=cat_dim_action
                    )
                    laser_action = np.concatenate(
                        (laser_action, exit_laser_action), axis=cat_dim_action
                    )

                    if random_agent:
                        random_exit_idx = random.randint(1, agent.policy_net.num_ee + 1)
                        exit_agent_state = state_batch_tensor[-2].unsqueeze(0)
                        (
                            exit_move_action,
                            exit_laser_action,
                            conf,
                            exit,
                            cost,
                        ) = agent.act(
                            exit_agent_state,
                            num_agents=1,
                            eval_all_exit=True,
                            eval_exit_point=random_exit_idx,
                        )

                        exits.append(exit)
                        confs = torch.cat((confs, conf), dim=cat_dim_tensors)
                        move_action = np.concatenate(
                            (move_action, exit_move_action), axis=cat_dim_action
                        )
                        laser_action = np.concatenate(
                            (laser_action, exit_laser_action), axis=cat_dim_action
                        )

                    exit_agent_state = state_batch_tensor[-1].unsqueeze(0)
                    (
                        exit_move_action,
                        exit_laser_action,
                        conf,
                        exit,
                        cost,
                    ) = agent.act(
                        exit_agent_state,
                        num_agents=1,
                        eval_agent=True,
                    )

                    exits.append(exit)
                    confs = torch.cat((confs, conf), dim=cat_dim_tensors)
                    move_action = np.concatenate(
                        (move_action, exit_move_action), axis=cat_dim_action
                    )
                    laser_action = np.concatenate(
                        (laser_action, exit_laser_action), axis=cat_dim_action
                    )

                    confs = confs.squeeze()
                    exits = torch.tensor(exits).unsqueeze(1)

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, _ = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id

                    if all_agents_active:
                        agents_to_act = agents_need_action
                        agent_ids_to_print = None
                    else:
                        agents_to_act = [active_agent_id[team_idx]]
                        agent_ids_to_print = agents_to_act

                    for agent_id in agents_to_act:
                        agent_move_action = move_action[agent_id, ...]
                        agent_laser_action = laser_action[agent_id, ...]

                        exit = exits[agent_id]
                        conf = confs[agent_id]
                        eval_agents[team][agent_id]["move_actions"] += agent_move_action
                        eval_agents[team][agent_id]["exit_points"][exit] += 1
                        eval_agents[team][agent_id]["agent_confs"][exit].append(
                            conf.detach().clone()
                        )

                        env.set_action_for_agent(
                            team,
                            agent_id,
                            ActionTuple(agent_move_action, agent_laser_action),
                        )

                env.step()

                for _, team in enumerate(team_name_list):
                    decision_steps, terminal_steps = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    if all_agents_active:
                        agents_to_act = agents_need_action
                        agent_ids_to_print = None
                    else:
                        agents_to_act = [active_agent_id[team_idx]]
                        agent_ids_to_print = agents_to_act

                    for agent_id in agents_to_act:
                        agent_dict = eval_agents[team][agent_id]
                        agent_obs = decision_steps[agent_id].obs
                        next_state = (
                            get_grid_based_perception(agent_obs).detach().clone()
                        )
                        state = next_state
                        state_batch_tensor[agent_id, ...] = state

                        agent_reward = decision_steps[agent_id].reward
                        if agent_reward == -4.0:
                            agent_dict["bad_food"] += 1

                        if agent_reward > 0.0:
                            agent_dict["good_food"] += 1

                        if agent_reward == -1.0:
                            agent_dict["wall_hit"] += 1

                        if agent_reward < -4.0:
                            agent_dict["bad_food"] += 1
                            agent_dict["wall_hit"] += 1

                        if float(agent_reward) != 0.0:
                            agent_dict["episode_score"] += agent_reward

                    terminated_agent_ids = terminal_steps.agent_id
                    done = True if len(terminated_agent_ids) > 0 else False
                    episode_done = done

            eval_scores_all_agents = extract_scores_for_all_agents(
                eval_agents, active_agents=agent_ids_to_print, flatten=True
            )
            mean_score = np.mean(eval_scores_all_agents)

            if current_episode is not None:
                print(
                    f"[EVAL] Mean performance on policy net after {current_episode} episodes: {mean_score:.2f}"
                )
            else:
                print(
                    f"[EVAL] Mean performance on trained policy net episodes: {mean_score:.2f}"
                )

            if not all_agents_active:
                extract_exit_points_from_agents(
                    eval_agents,
                    active_agent_id=active_agent_id,
                    include_reward=True,
                    include_food_info=True,
                    include_wall_info=True,
                    include_action_distribution=True,
                    print_out=True,
                )
            else:
                extract_exit_points_from_agents(
                    eval_agents,
                    include_reward=True,
                    include_food_info=True,
                    include_wall_info=True,
                    include_action_distribution=True,
                    print_out=True,
                )

            all_episode_values = extract_exit_points_from_agents(
                eval_agents, print_out=False
            )
            exit_points.append(all_episode_values[0])
            rewards.append(all_episode_values[1])
            good_food.append(all_episode_values[2])
            bad_food.append(all_episode_values[3])
            wall_hit.append(all_episode_values[4])
            action_dist.append(all_episode_values[5])

        if was_in_training:
            agent.policy_net.train()

    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        print(ex)
        print("-" * 100)
        print("Exception has occured !!")
        print("Evaluation was interrupted.")
        print("-" * 100)
        env.close()
    finally:
        if results_directory is not None:
            exit_threshold = agent.policy_net.exit_threshold
            save_list_to_json(exit_points, f"{results_directory}/exit_points.json")
            save_list_to_json(rewards, f"{results_directory}/rewards.json")
            save_list_to_json(good_food, f"{results_directory}/good_food.json")
            save_list_to_json(bad_food, f"{results_directory}/bad_food.json")
            save_list_to_json(wall_hit, f"{results_directory}/wall_hit.json")
            save_list_to_json(action_dist, f"{results_directory}/action_dist.json")
            save_list_to_json(exit_threshold, f"{results_directory}/exit_threshold.json")

        # if eval_agents:
        #     eval_scores_all_agents = [
        #         team_info["episode_score"] for team_info in eval_agents.values()
        #     ]
        # else:
        #     eval_scores_all_agents = None

    # return eval_scores_all_agents
