# The given format dictionary with random episode keys and varying lengths
eval_agents = {
    "GridFoodCollector?team=1": {
        0: {"episode_score": 5},
        7: {"episode_score": 8},
        11: {"episode_score": 9},
    },
}


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


flat_scores = extract_scores_for_all_agents(
    eval_agents, all_agents_active=False, flatten=True
)
print("1D List:")
print(flat_scores)

score_matrix = extract_scores_for_all_agents(eval_agents)
print("2D List:")
for row in score_matrix:
    print(row)
