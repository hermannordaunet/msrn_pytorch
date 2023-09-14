import torch

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


def extract_scores_for_all_agents(
    eval_agents: dict, all_agents_active=True, flatten=False
):
    # Extract the team keys from the eval_agents
    team_keys = list(eval_agents.keys())

    # Extract all episode keys from all teams' dictionaries
    agent_keys = list()
    for team_key in team_keys:
        agents_in_team = list(eval_agents[team_key].keys())

        if all_agents_active:
            agent_keys.append(tuple(agents_in_team))

        else:
            agent_keys.append([agents_in_team[-1]])

    if flatten:
        # Flatten the scores into a 1D list
        scores = []
        for team_id, team_key in enumerate(team_keys):
            for agent_key in agent_keys[team_id]:
                scores.append(eval_agents[team_key][agent_key]["episode_score"])
    else:
        # Initialize a 2D list with zeros based on the number of teams and episodes
        num_teams = len(team_keys)
        scores = [None] * num_teams

        # Iterate through teams and episodes to fill the scores
        for i, team_key in enumerate(team_keys):
            scores[i] = [0] * len(agent_keys[i])
            for j, episode_key in enumerate(agent_keys[i]):
                if episode_key in eval_agents[team_key]:
                    scores[i][j] = eval_agents[team_key][episode_key]["episode_score"]

    return scores


def extract_one_agent_each_team(eval_agents: dict):
    team_keys = list(eval_agents.keys())
    one_agent_keys = list()
    for team_key in team_keys:
        agents_in_team = list(eval_agents[team_key].keys())
        one_agent_keys.append(agents_in_team[-1])

    return one_agent_keys


def print_exit_points_from_agents(eval_agents: dict, active_agent_id=None):
    for team, team_data in eval_agents.items():
        agent_ids = team_data.keys()
        if active_agent_id is None:
            agents_to_print = team_data.keys()
        else:
            agents_to_print = active_agent_id

        for agent_id in agent_ids:
            if agent_id in agents_to_print:
                exit_points = eval_agents[team][agent_id]["exit_points"]
                print(f"Agent ID: {agent_id}, Exit Points: {exit_points}")


def evaluate_trained_model(env, agent, config, current_episode, verbose=False):
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

    state_size = agent.model_param["input_size"]
    state_batch_tensor = torch.zeros((num_total_agents, *state_size))

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
                eval_agents[team] = {}

                decision_steps, _ = env.get_steps(team)
                agents_need_action = decision_steps.agent_id
                for agent_id in agents_need_action:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs)
                    state_batch_tensor[agent_id, ...] = state

                    eval_agents[team][agent_id] = {
                        "episode_score": 0,
                    }

            episode_done = False

            if not all_agents_active:
                active_agent_id = extract_one_agent_each_team(eval_agents)

            while not episode_done:
                act = agent.act(state_batch_tensor, num_agents=num_total_agents)
                move_action, laser_action = act

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, _ = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    if all_agents_active:
                        for agent_id in agents_need_action:
                            agent_move_action = move_action[agent_id, ...]
                            agent_laser_action = laser_action[agent_id, ...]
                            env.set_action_for_agent(
                                team,
                                agent_id,
                                ActionTuple(agent_move_action, agent_laser_action),
                            )
                    else:
                        agent_id = active_agent_id[team_idx]
                        if agent_id in agents_need_action:
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
                    if all_agents_active:
                        for agent_id in agents_need_action:
                            agent_obs = decision_steps[agent_id].obs
                            next_state = get_grid_based_perception(agent_obs)
                            state = next_state
                            state_batch_tensor[agent_id, ...] = state

                            agent_reward = decision_steps[agent_id].reward
                            eval_agents[team][agent_id]["episode_score"] += agent_reward
                    else:
                        agent_id = active_agent_id[team_idx]
                        if agent_id in agents_need_action:
                            agent_obs = decision_steps[agent_id].obs
                            next_state = get_grid_based_perception(agent_obs)
                            state = next_state
                            state_batch_tensor[agent_id, ...] = state

                            agent_reward = decision_steps[agent_id].reward
                            eval_agents[team][agent_id]["episode_score"] += agent_reward

                    terminated_agent_ids = terminal_steps.agent_id
                    done = True if len(terminated_agent_ids) > 0 else False
                    episode_done = done

            eval_scores_all_agents = extract_scores_for_all_agents(
                eval_agents, all_agents_active=all_agents_active, flatten=True
            )
            mean_score = np.mean(eval_scores_all_agents)

            print(
                f"[INFO] Mean performance on policy net after {current_episode} episodes: {mean_score}"
            )

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
    # finally:
    # if eval_agents:
    #     eval_scores_all_agents = [
    #         team_info["episode_score"] for team_info in eval_agents.values()
    #     ]
    # else:
    #     eval_scores_all_agents = None

    # return eval_scores_all_agents
