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

def get_scores_of_all_agents(eval_agents: dict(), flatten=False):
    if flatten:
        # Flatten the scores into a 1D list
        scores = [eval_agents[team][j]['episode_score'] for team in ['team0', 'team1', 'team2'] for j in range(4)]
    else:
        # Initialize a 3x4 list with zeros
        scores = [[0] * 4 for _ in range(3)]

        # Iterate through teams and episodes to fill the score_matrix
        for i, team in enumerate(['team0', 'team1', 'team2']):
            for j in range(4):
                scores[i][j] = eval_agents[team][j]['episode_score']

    return scores



def evaluate_trained_model(env, agent, config, current_episode, verbose=False):
    num_eval_episodes = config["eval"]["episodes"]

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

                        agent_reward = decision_steps[agent_id].reward
                        eval_agents[team][agent_id]["episode_score"] += agent_reward

                    terminated_agent_ids = terminal_steps.agent_id
                    done = True if len(terminated_agent_ids) > 0 else False
                    episode_done = done
            
            eval_scores_all_agents = get_scores_of_all_agents(eval_agents, flatten=True)
            mean_score = np.mean(eval_scores_all_agents)
            
            print(f"[INFO] Mean performance on policy net after {current_episode} episodes: {mean_score}")
        
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
