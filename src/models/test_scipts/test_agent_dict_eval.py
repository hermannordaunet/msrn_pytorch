def print_exit_points(format_dict, agent_ids_list=None):
    for team, team_data in format_dict.items():
        agent_ids = team_data.keys()
        if agent_ids_list is None:
            agents_to_print = team_data.keys()
        else:
            agents_to_print = agent_ids_list

        for agent_id in agent_ids:
            if agent_id in agents_to_print:
                exit_points = format_dict[team][agent_id]["exit_points"]
                print(f"Agent ID: {agent_id}, Exit Points: {exit_points}")


# Example usage:
format_dict = {
    "GridFoodCollector?team=1": {
        0: {"episode_score": 5, "exit_points": [0, 15, 30]},
        7: {"episode_score": 8, "exit_points": [0, 15, 30]},
        11: {"episode_score": 9, "exit_points": [0, 15, 30]},
    },
    "GridFoodCollector?team=0": {
        1: {"episode_score": 2, "exit_points": [0, 15, 30]},
        5: {"episode_score": 3, "exit_points": [0, 15, 30]},
        10: {"episode_score": 4, "exit_points": [0, 15, 30]},
    },
    "GridFoodCollector?team=3": {
        2: {"episode_score": 1, "exit_points": [0, 15, 30]},
        6: {"episode_score": 7, "exit_points": [0, 15, 30]},
    },
    "GridFoodCollector?team=2": {
        3: {"episode_score": 6, "exit_points": [0, 15, 30]},
        4: {"episode_score": 0,"exit_points": [0, 15, 30]},
        8: {"episode_score": 10, "exit_points": [0, 15, 30]},
        12: {"episode_score": 11, "exit_points": [0, 15, 30]},
        19: {"episode_score": 12, "exit_points": [0, 15, 30]},
    },
}

print_exit_points(format_dict, [7, 11, 12])
