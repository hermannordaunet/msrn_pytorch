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

from src.models.evaluate_unity import (
    extract_one_agent_each_team,
    extract_scores_for_all_agents,
    extract_exit_points_from_agents,
)


def visualize_trained_model(env, agent, config, verbose=False):
    num_visual_episodes = config["visualize"]["episodes"]

    team_name_list = list(env.behavior_specs.keys())
    num_teams = len(team_name_list)
    decision_steps, _ = env.get_steps(team_name_list[-1])

    num_agents_on_teams = len(decision_steps.agent_id)
    num_total_agents = num_teams * num_agents_on_teams
    all_agents_active = config["visualize"]["all_agents_active"]

    state_size = agent.model_param["input_size"]
    state_batch_tensor = torch.zeros((num_total_agents, *state_size))

    try:
        if verbose:
            print(
                f"[INFO] Number of parallell environments during visualization: {num_teams}"
            )

        for episode in range(1, num_visual_episodes + 1):
            visual_agents = dict()

            if verbose:
                print(f"\nEpisode {episode}/{num_visual_episodes} started")

            env.reset()

            for _, team in enumerate(team_name_list):
                visual_agents[team] = {}

                decision_steps, _ = env.get_steps(team)
                agents_need_action = decision_steps.agent_id
                for agent_id in agents_need_action:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs)
                    state_batch_tensor[agent_id, ...] = state

                    visual_agents[team][agent_id] = {
                        "bad_food": 0,
                        "good_food": 0,
                        "episode_score": 0,
                        "exit_points": [0] * (agent.policy_net.num_ee + 1),
                        "agent_confs": [[] for _ in range(agent.policy_net.num_ee + 1)],
                    }

            episode_done = False

            if not all_agents_active:
                active_agent_id = extract_one_agent_each_team(visual_agents)
            else:
                active_agent_id = None
                
            while not episode_done:
                move_action, laser_action, confs, exits, costs = agent.act(
                    state_batch_tensor, num_agents=num_total_agents, eval_agent=True
                )

                for team_idx, team in enumerate(team_name_list):
                    decision_steps, _ = env.get_steps(team)
                    agents_need_action = decision_steps.agent_id
                    if all_agents_active:
                        for agent_id in agents_need_action:
                            agent_move_action = move_action[agent_id, ...]
                            agent_laser_action = laser_action[agent_id, ...]

                            exit = exits[agent_id]
                            conf = confs[agent_id]
                            visual_agents[team][agent_id]["exit_points"][exit] += 1
                            visual_agents[team][agent_id]["agent_confs"][exit].append(
                                conf.detach().clone()
                            )

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

                            exit = exits[agent_id]
                            conf = confs[agent_id]
                            visual_agents[team][agent_id]["exit_points"][exit] += 1
                            visual_agents[team][agent_id]["agent_confs"][exit].append(
                                conf
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
                        for agent_id in agents_need_action:
                            agent_dict = visual_agents[team][agent_id]
                            agent_obs = decision_steps[agent_id].obs
                            next_state = get_grid_based_perception(agent_obs)
                            state = next_state
                            state_batch_tensor[agent_id, ...] = state

                            agent_reward = decision_steps[agent_id].reward
                            if agent_reward < 0.0:
                                visual_agents[team][agent_id]["bad_food"] += 1

                            if agent_reward > 0.0:
                                visual_agents[team][agent_id]["good_food"] += 1

                            if float(agent_reward) != 0.0:
                                agent_dict["episode_score"] += agent_reward

                    else:
                        agent_id = active_agent_id[team_idx]
                        if agent_id in agents_need_action:
                            agent_dict = visual_agents[team][agent_id]
                            agent_obs = decision_steps[agent_id].obs
                            next_state = get_grid_based_perception(agent_obs)
                            state = next_state
                            state_batch_tensor[agent_id, ...] = state

                            agent_reward = decision_steps[agent_id].reward
                            if agent_reward < 0.0:
                                visual_agents[team][agent_id]["bad_food"] += 1

                            if agent_reward > 0.0:
                                visual_agents[team][agent_id]["good_food"] += 1

                            if float(agent_reward) != 0.0:
                                agent_dict["episode_score"] += agent_reward

                    terminated_agent_ids = terminal_steps.agent_id
                    done = True if len(terminated_agent_ids) > 0 else False
                    episode_done = done

            visual_scores_all_agents = extract_scores_for_all_agents(
                visual_agents, all_agents_active=all_agents_active, flatten=True
            )
            mean_score = np.mean(visual_scores_all_agents)

            print(f"[VISUALIZE] Mean performance on policy net: {mean_score}")

            if not all_agents_active:
                extract_exit_points_from_agents(
                    visual_agents,
                    active_agent_id=active_agent_id,
                    include_reward=True,
                    include_food_info=True,
                    mode="VISUALIZE",
                    print_out=True,
                )
            else:
                extract_exit_points_from_agents(
                    visual_agents,
                    include_reward=True,
                    include_food_info=True,
                    mode="VISUALIZE",
                    print_out=True,
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
        print("Visualizing was interrupted.")
        print("-" * 100)
        env.close()
