import torch

# unity imports
from mlagents_envs.base_env import ActionTuple

from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)

# Local imports
from src.models.utils.data_utils import get_grid_based_perception


def visualize_trained_model(env, agent, config, verbose=False):
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

            env.reset()

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
                move_action, laser_action, confs, exits, costs = act

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