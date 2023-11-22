import json
import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd

# TODO: Add some plotting functions into this.
# Training loss and score


def set_size(width="thesis", fraction=1, subplots=(1, 1), golden_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 418.25368
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if not golden_ratio:
        golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_reward_for_each_agent(
    scores: list(),
    plot_type="box",
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
    exit_threshold=None,
):
    df = pd.DataFrame(scores)

    df_melted = df.melt(
        var_name="Agent",
        value_name="Reward",
    )

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()
    if plot_type == "box":
        plot = sns.boxplot(
            x="Agent",
            y="Reward",
            data=df_melted,
            notch=True,
            flierprops={"marker": "o"},
        )
    else:
        plot = sns.violinplot(x="Agent", y="Reward", data=df_melted, errorbar="ci", estimator="mean")

    plt.ylim((-25, 30))
    plt.suptitle("Rewards for each agent type", fontsize=12)

    if exit_threshold is not None:
        plt.title(f"MSRN exit thresholds: {exit_threshold}", fontsize=10, ha="center")

    plt.xlabel("Agents")
    plt.ylabel("Reward")

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/rewards_{plot_type}.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()


def plot_action_distribution(
    action_dist,
    labels,
    result_dir="./",
    rotate_labels=False,
):
    num_episodes, num_agents, _ = action_dist.shape
    numpy_action_dist = action_dist.numpy()

    # Initialize columns
    data = {"agent_id": [], "Forward": [], "Side motion": [], "Rotation": []}

    # Define agent types
    agent_types = ["exit 1", "exit 2", "exit 3", "exit 4", "exit 5", "full", "random", "MSRN"]

    # Populate the columns
    for i in range(num_episodes):  # Loop through episodes
        for j in range(num_agents):  # Loop through agents
            data["agent_id"].append(agent_types[j])
            data["Forward"].append(numpy_action_dist[i, j, 0])
            data["Side motion"].append(numpy_action_dist[i, j, 1])
            data["Rotation"].append(numpy_action_dist[i, j, 2])

    # Create DataFrame
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='agent_id', var_name='motion_type', value_name='value')

    plt.figure(figsize=set_size(golden_ratio=0.8))
    sns.set_theme(style="darkgrid")
    sns.despine()
    plot = sns.barplot(data=df_melted, x="motion_type", y="value", hue="agent_id", errorbar="ci", estimator="mean", capsize=0.05, errwidth=0.8)

    plt.title("Mean Action Counts per Agent with Standard Deviation")
    plt.xlabel("")
    plot.xaxis.set_ticks_position("none")
    plt.ylabel("Mean Count")
    plt.legend()

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/action_dist.pdf", format="pdf", bbox_inches="tight"
    )

    plt.close()


def plot_exit_distribution(
    exit_dist: list(),
    agent_type="msrn",
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=False,
    exit_threshold=None,
):
    if agent_type == "msrn":
        last_agent_data = exit_dist[:, -1, :]
        agent_string = "MSRN"
    elif agent_type == "random":
        last_agent_data = exit_dist[:, -2, :]
        agent_string = "Random"
    else:
        print("No supported agent type provided. Plotting MSRN exit distribution.")
    number_of_episodes = exit_dist.shape[0]
    number_of_exits = exit_dist.shape[-1]
    number_of_agents = exit_dist.shape[1]

    exit_numbers = np.tile(
        np.arange(number_of_exits), (number_of_episodes, 1)
    )  # Repeating exit numbers for each episode
    episode_numbers = np.repeat(
        np.arange(1, number_of_episodes + 1), number_of_exits
    )  # Repeating each episode number 4 times

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Episode": episode_numbers,
            "Exit": exit_numbers.flatten(),
            "Count": last_agent_data.flatten(),
        }
    )

    # Now create the barplot using Seaborn
    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    plot = sns.barplot(data=df, x="Exit", y="Count", estimator="mean", errorbar="ci", capsize=0.15, errwidth=1
    )
    plt.suptitle(
        f"Average Exit Count for {agent_string} agent with Standard Deviation", fontsize=12
    )
    sns.despine()

    if exit_threshold is not None:
        plt.title(f"MSRN exit thresholds: {exit_threshold}", fontsize=10)

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/exit_dist_{agent_type}.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()


def get_macs_for_agent(exit_dist, exit_macs):
    number_of_steps = torch.sum(exit_dist[0, 0, 0])
    result = (
        torch.einsum("ijk,k->ij", exit_dist.type(torch.float32), exit_macs)
        / number_of_steps
    )

    return result


def get_cost_pareto(exit_dist):
    num_exits = exit_dist.shape[-1]

    if num_exits == 4:
        cost_list = [21.22, 34.36, 42.63, 86.84]
    elif num_exits == 5:
        cost_list = [34.38, 64.83, 86.09, 107.35, 170.45]
    elif num_exits == 6:
        cost_list = [34.38, 64.83, 86.09, 107.35, 117.98, 170.45]

        # cost_list = [28.63, 64.83, 86.09, 117.98, 151.56, 170.45]
    else:
        print("No exit cost_list for this amount of exits.")
        exit()

    cost_list = torch.tensor(cost_list)
    return cost_list

def get_cost_linear(exit_dist):
    num_exits = exit_dist.shape[-1]

    if num_exits == 6:
        cost_list = [28.63, 64.83, 86.09, 117.98, 151.56, 170.45]

    else:
        print("No exit cost_list for this amount of exits.")
        exit()

    cost_list = torch.tensor(cost_list)
    return cost_list


def get_mean_df_score(scores):
    df = pd.DataFrame(scores)
    df_mean = df.mean()

    return df_mean


def get_all_mean_scores(scores):
    mean_scores = get_mean_df_score(scores)
    msrn_mean = mean_scores.iloc[-1]
    random_mean = mean_scores.iloc[-2]
    backbone_mean = mean_scores.iloc[-3]

    return msrn_mean, random_mean, backbone_mean


def get_all_mean_macs(exit_dist):
    cost_list = get_cost_pareto(exit_dist)
    macs_tensor = get_macs_for_agent(exit_dist, cost_list)
    mean_values = macs_tensor.mean(dim=0)

    msrn_mmacs = mean_values[-1]
    random_mmacs = mean_values[-2]
    backbone_mmacs = mean_values[-3]

    return msrn_mmacs, random_mmacs, backbone_mmacs

def get_all_macs(exit_dist):
    cost_list = get_cost_pareto(exit_dist)
    macs_tensor = get_macs_for_agent(exit_dist, cost_list)

    msrn_mmacs = macs_tensor[:, -1]
    random_mmacs = macs_tensor[:, -2]
    backbone_mmacs = macs_tensor[:, -3]

    return msrn_mmacs, random_mmacs, backbone_mmacs


def plot_macs_for_agent(
    exit_dist: list(),
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
    exit_threshold=None,
    timestamp=None,
):
    
    if timestamp.startswith("1700306774"):
        cost_list = get_cost_linear(exit_dist)
    else:
        cost_list = get_cost_pareto(exit_dist)

    macs_tensor = get_macs_for_agent(exit_dist, cost_list)

    # Now create the barplot using Seaborn
    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    plot = sns.barplot(
        data=macs_tensor, estimator="mean", errorbar="ci", capsize=0.15, errwidth=1
    )
    plt.suptitle(
        "MACs for each agent (95% CI)", fontsize=12
    )
    plt.ylabel("Millions of MACs (MMACs)")
    plt.xlabel("Agent type")

    if exit_threshold is not None:
        plt.title(f"MSRN exit thresholds: {exit_threshold}", fontsize=10)

    num_static_mac_agents = macs_tensor.shape[-1] - 2
    num_lines_to_remove = num_static_mac_agents * 3 # each error bar has 3 lines (cap and bar), so multiply by 3

    lines = plot.get_lines()
    for i in range(num_lines_to_remove):
        lines[i].set_visible(False)


    sns.despine()

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(f"{result_dir}/macs.pdf", format="pdf", bbox_inches="tight")
    plt.close()


# save scores plot
def plot_scores_from_list(scores: list(), labels=None, env_name="", result_dir="./"):
    scores = np.array(scores).squeeze()

    if scores.ndim > 1:
        x_values = np.arange(1, scores[-1].size + 1)
    else:
        x_values = np.arange(1, scores.size + 1)

    if labels:
        if scores.ndim > 1:
            if scores.shape[0] > len(labels):
                print("Need to provide enough labels to get legend on the plot")
                labels = None

    df = pd.DataFrame(scores.transpose(), columns=labels, index=x_values)

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df)
    sns.despine()
    plt.title(f"{env_name} agent reward")
    plt.ylabel("Reward")
    plt.xlabel("Episode")

    if labels:
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{result_dir}/{env_name}_scores.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()


def plot_scores_from_nested_list(
    scores: list(), labels=None, env_name="FoodCollector", result_dir="./"
):
    numpy_scores = np.array(scores)
    num_episodes, num_agents = scores.shape

    episodes = range(1, num_episodes+1, )
    # You should now create a list to index your data based on thresholds
    df_episode = np.array([[episode]*num_agents for episode in episodes]).reshape(-1, 1)   

    reshaped_scores = scores.view(-1, 1).numpy()
    scores_df = pd.DataFrame(np.concatenate((df_episode, reshaped_scores), axis=1), columns=["Episode", "Score"])

    if labels:
        if len(labels) == 1:
            labels = labels[-1]

    # Create the line plot with error bars
    plt.figure(figsize=set_size())
    # Set Seaborn style
    sns.set(style="darkgrid")
    plot = sns.lineplot(data=scores_df, x="Episode", y="Score", estimator="mean", errorbar="ci", label=labels)
    sns.despine()
    plt.title(f"{env_name} Reward")
    plt.ylabel("Mean Reward")
    plt.xlabel("Episode")

    if labels:
        plt.legend()

    # plt.tight_layout()
    plt.savefig(
        f"{result_dir}/{env_name}_scores_parallell_agents.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_loss_from_list(
    losses: list(), labels=None, env_name="", result_dir="./", loss_type="Q-value"
):
    losses = np.array(losses).squeeze()

    if losses.ndim > 1:
        x_values = np.arange(1, losses[-1].size + 1)
    else:
        x_values = np.arange(1, losses.size + 1)

    if labels:
        if losses.ndim > 1:
            if losses.shape[0] > len(labels):
                print("Need to provide enough labels to get legend on the plot")
                labels = None

    df = pd.DataFrame(losses.transpose(), columns=labels, index=x_values)

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df)
    sns.despine()
    plt.title(f"{env_name}: {loss_type} loss")
    plt.ylabel("Loss")
    plt.xlabel("Episode")

    if labels:
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{result_dir}/{env_name}_{loss_type}_losses.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_grid_based_perception(
    image_tensor, team_id=None, title=None, save_figure=True, **kwargs
):
    image_tensor = image_tensor.squeeze().cpu()
    if len(image_tensor.shape) == 4:
        num_rows, num_cols, _, _ = image_tensor.shape
    else:
        num_rows = 1
        num_cols, _, _ = image_tensor.shape

    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=set_size(subplots=(num_rows, num_cols), golden_ratio=5),
    )

    labels = ["food", "agent", "wall", "badFood", "frozenAgent"]

    for row in range(num_rows):
        for col in range(num_cols):
            if num_rows > 1 and num_cols > 1:
                cur_ax = ax[row, col]
                cur_ax.imshow(image_tensor[row, col, ...], interpolation=None)
            else:
                cur_ax = ax[col]
                cur_ax.imshow(image_tensor[col, ...], interpolation=None)

            if row == 0:
                cur_ax.set_title(labels[col])

            cur_ax.axis("off")

    if title:
        fig.suptitle(str(title))
    else:
        if team_id:
            fig.suptitle(f"Agent observation: {team_id}")

    plt.tight_layout()

    if save_figure:
        plt.savefig(f"grid-based-observation.pdf", format="pdf", bbox_inches="tight")

    plt.close()
    # plt.show(**kwargs)


def load_json_as_list(file_path):
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data


def create_dynamic_list(number_of_early_exits, with_random=True):
    # The base labels that are always included
    if with_random:
        base_labels = ["Full", "Random", "MSRN"]
    else:
        base_labels = ["Full", "MSRN"]
    # Create the exit labels based on the number_of_exits variable
    exit_labels = [f"Exit {i}" for i in range(1, number_of_early_exits + 1)]

    # Combine the exit labels with the base labels
    dynamic_list = exit_labels + base_labels

    return dynamic_list

def plot_split_violin_plot_18_vs_34(msrn_rewards, save_figure=True, result_dir="./"):

    data = msrn_rewards.numpy() if not isinstance(msrn_rewards, np.ndarray) else msrn_rewards

    # Reshape data: Each row is an episode, columns are 'AgentType' and 'Reward'
    # Let's assume agent types are 0 and 1, you can replace these with actual names if available

    agent_types = np.repeat(["ResNet18", "ResNet34"], data.shape[1])

    episodes = np.tile(np.arange(data.shape[1]), 2)
    rewards = data.flatten()
    agent_classes = np.repeat([" ", " "], data.shape[1])

    df = pd.DataFrame({'AgentType': agent_types, 'Episode': episodes, 'Reward': rewards, "AgentClasses": agent_classes})

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()

    plot = sns.violinplot(data=df, y="AgentClasses", x="Reward", hue="AgentType", split=True, gap=.075, inner="box")

    plt.xlabel("Reward")
    plt.ylabel("")

    plt.tight_layout()

    if save_figure:
        plt.savefig(f"{result_dir}/18_vs_34_violin_tilted.pdf", format="pdf", bbox_inches="tight")

    plt.close()

def plot_split_violin_plot_pareto_vs_linear(msrn_rewards, save_figure=True, result_dir="./"):

    data = msrn_rewards.numpy() if not isinstance(msrn_rewards, np.ndarray) else msrn_rewards

    # Reshape data: Each row is an episode, columns are 'AgentType' and 'Reward'
    # Let's assume agent types are 0 and 1, you can replace these with actual names if available

    agent_types = np.repeat(["Pareto", "Linear"], data.shape[1])

    episodes = np.tile(np.arange(data.shape[1]), 2)
    rewards = data.flatten()
    agent_classes = np.repeat([" ", " "], data.shape[1])

    df = pd.DataFrame({'AgentType': agent_types, 'Episode': episodes, 'Reward': rewards, "AgentClasses": agent_classes})

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()

    plot = sns.violinplot(data=df, y="AgentClasses", x="Reward", hue="AgentType", split=True, gap=.075, inner="box")

    plt.xlabel("Reward")
    plt.ylabel("")

    plt.tight_layout()

    if save_figure:
        plt.savefig(f"{result_dir}/pareto_vs_linear_violin_tilted.pdf", format="pdf", bbox_inches="tight")

    plt.close()


def plot_split_violin_plot_comp_vs_no_comp(msrn_rewards, save_figure=True, result_dir="./"):

    data = msrn_rewards.numpy() if not isinstance(msrn_rewards, np.ndarray) else msrn_rewards

    # Reshape data: Each row is an episode, columns are 'AgentType' and 'Reward'
    # Let's assume agent types are 0 and 1, you can replace these with actual names if available

    agent_types = np.repeat(["Comp", "Non-comp"], data.shape[1])

    episodes = np.tile(np.arange(data.shape[1]), 2)
    rewards = data.flatten()
    agent_classes = np.repeat([" ", " "], data.shape[1])

    df = pd.DataFrame({'Agent Type': agent_types, 'Episode': episodes, 'Reward': rewards, "AgentClasses": agent_classes})

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()

    plot = sns.violinplot(data=df, y="AgentClasses", x="Reward", hue="Agent Type", split=True, gap=.075, inner="box")

    plt.xlabel("Reward")
    plt.ylabel("")

    plt.tight_layout()

    if save_figure:
        plt.savefig(f"{result_dir}/comp_vs_noncomp_violin_tilted.pdf", format="pdf", bbox_inches="tight")

    plt.close()

def plot_split_violin_plot(msrn_rewards, save_figure=True, result_dir="./", training_type="regular"):

    # Assuming your tensor is named 'tensor' and is convertible to a numpy array
    data = msrn_rewards.numpy() if not isinstance(msrn_rewards, np.ndarray) else msrn_rewards

    # Reshape data: Each row is an episode, columns are 'AgentType' and 'Reward'
    # Let's assume agent types are 0 and 1, you can replace these with actual names if available

    if training_type == "regular":
        agent_types = np.repeat(["Regular", "MSRN"], data.shape[1])
    elif training_type == "e2e":
        agent_types = np.repeat(["End-to-End", "Element-wise"], data.shape[1])

    episodes = np.tile(np.arange(data.shape[1]), 2)
    rewards = data.flatten()
    agent_classes = np.repeat([" ", " "], data.shape[1])

    df = pd.DataFrame({'AgentType': agent_types, 'Episode': episodes, 'Reward': rewards, "AgentClasses": agent_classes})

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()

    plot = sns.violinplot(data=df, y="AgentClasses", x="Reward", hue="AgentType", split=True, gap=.075, inner="box")

    plt.xlabel("Reward")
    if training_type == "e2e":
        plt.ylabel("Training scheme")
    else:
        plt.ylabel("")

    plt.tight_layout()

    if save_figure:
        plt.savefig(f"{result_dir}/training_{training_type}_violin_tilted.pdf", format="pdf", bbox_inches="tight")

    plt.close()




def plot_reward_vs_macs(
    msrn_rewards,
    msrn_macs,
    random_macs_count=None,
    random_reward=None,
    backbone_reward=None,
    thresholds=None,
    rotate_labels=False,
    result_dir="./",
):
    figure_fontsize = 12

    thresholds = list(thresholds)
    # You should now create a list to index your data based on thresholds
    df_thresholds = np.array([[threshold]*250 for threshold in thresholds]).reshape(-1, 1)   

    reshaped_msrn_rewards = msrn_rewards.view(-1, 1).numpy()
    msrn_rewards_df = pd.DataFrame(np.concatenate((df_thresholds, reshaped_msrn_rewards), axis=1), columns=["Threshold", "Reward"])
    msrn_rewards_T = torch.t(msrn_rewards)
    # msrn_rewards_df = pd.DataFrame(msrn_rewards_T)
    # print(msrn_rewards_df)

    plt.figure(figsize=set_size(golden_ratio=1))
    sns.set_theme(style="whitegrid")
    plt.title("Reward vs MACs")

    # Mean reward
    mean_msrn_rewards = torch.mean(msrn_rewards_T, axis=0)
    min_msrn_rewards = torch.min(msrn_rewards_T, axis=0)[0]
    max_msrn_rewards = torch.max(msrn_rewards_T, axis=0)[0]

    # ax1 = sns.lineplot(
    #     x=thresholds, y=mean_msrn_rewards, sort=False, label="MSRN Mean Reward"
    # )

    ax1 = sns.lineplot(
        data=msrn_rewards_df, 
        x="Threshold", 
        y="Reward", 
        estimator="mean", 
        errorbar="ci", 
        label="MSRN Mean Reward"
    )
    # plt.fill_between(thresholds, min_msrn_rewards, max_msrn_rewards, alpha=0.1)
    ax1.set_ylabel("Mean Reward")
    # ax1.set_ylim(0, 25)
    # ax1.legend(["MSRN Mean Reward"])
    # ax1.yaxis.label.set_color(p1.get_color())
    # ax1.yaxis.label.set_fontsize(figure_fontsize)
    # ax1.tick_params(axis='y', colors=p1.get_color(), labelsize=figure_fontsize)

    # Mean MACs
    ax2 = ax1.twinx()

    reshaped_msrn_macs = msrn_macs.view(-1, 1).numpy()
    msrn_macs_df = pd.DataFrame(np.concatenate((df_thresholds, reshaped_msrn_macs), axis=1), columns=["Threshold", "MACs"])

    msrn_macs_T = torch.t(msrn_macs)
    # plot bar chart on axis #2
    mean_msrn_macs = torch.mean(msrn_macs_T, axis=0)
    min_msrn_macs = torch.min(msrn_macs_T, axis=0)[0]
    max_msrn_macs = torch.max(msrn_macs_T, axis=0)[0]
    sns.lineplot(
        data=msrn_macs_df, 
        x="Threshold", 
        y="MACs", 
        estimator="mean", 
        errorbar="sd", 
        label="MSRN Mean Macs",
        color="orange",
        linestyle="--",
        ax=ax2, # Pre-existing axes for the plot
    )
    ax2.invert_yaxis()
    # ax2.fill_between(thresholds, min_msrn_macs, max_msrn_macs, alpha=0.1, color="orange")
    # p2, = ax2.plot(thresholds, macs, color='orange', label='Mean MSRN MACs')
    ax2.grid(False)  # turn off grid #2
    ax2.set_ylabel("Mean Million of MACs (MMACs)")
    # ax2.set_ylim(0, 90)
    # ax2.legend(['Mean MSRN MACs'])
    # ax2.yaxis.label.set_color(p2.get_color())
    # ax2.yaxis.label.set_fontsize(figure_fontsize)
    # ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=figure_fontsize)


    if backbone_reward is not None:
        backbone_color = "green"
        backbone_reward_T = torch.t(backbone_reward)
        # Mean reward
        mean_mean_backbone_reward = torch.mean(backbone_reward)
        # max_mean_backbone_reward = torch.mean(mean_backbone_reward)
        # min_backbone_reward = torch.min(backbone_reward_T, axis=0)[0]
        # max_backbone_reward = torch.max(backbone_reward_T, axis=0)[0]

        # sns.lineplot(
        #     x=thresholds,
        #     y=mean_backbone_reward,
        #     sort=False,
        #     label="Backbone Mean Reward",
        #     alpha=0.4,
        #     ax=ax1,
        # )
        ax1.axhline(mean_mean_backbone_reward, color=backbone_color, alpha=0.4, label="Backbone Mean Reward")
        # plt.fill_between(thresholds, min_backbone_reward, max_backbone_reward, alpha=0.25)

    if random_reward is not None:
        random_reward_T = torch.t(random_reward)
        # Mean reward
        mean_random_reward = torch.mean(random_reward, axis=0)
        mean_mean_random_reward = torch.mean(random_reward)
        mean_random_reward_list = torch.full((len(thresholds),), mean_mean_random_reward)
        # min_random_reward = torch.min(random_reward_T, axis=0)[0]
        # max_random_reward = torch.max(random_reward_T, axis=0)[0]

        random_agent_color = "red"
        random_agent_alpha = 0.25
        random_agent_label = "Random Mean Reward"
        
        ax1.axhline(mean_mean_random_reward, color=random_agent_color, alpha=random_agent_alpha, label=random_agent_label)
        sns.lineplot(
            x=thresholds,
            y=mean_random_reward_list,
            sort=False,
            alpha=0,
            color=random_agent_color,
            ax=ax1,
        )

    # Add extra numbers on y-axis
    backbone_mmacs = 170.45
    # Get current yticks and labels
    yticks = list(ax2.get_yticks())
    ylabels = [str(label) for label in yticks]

    # Remove the tick to close to the random macs
    del yticks[4]
    del ylabels[4]

    # Add extra tick and label
    yticks.append(backbone_mmacs)
    ylabels.append(f"{backbone_mmacs:.2f} (a)")

    if random_macs_count is not None:
        reshaped_random_macs = random_macs_count.view(-1, 1).numpy()
        random_macs_df = pd.DataFrame(np.concatenate((df_thresholds, reshaped_random_macs), axis=1), columns=["Threshold", "MACs"])

        random_macs_count_T = torch.t(random_macs_count)
        # plot bar chart on axis #2
        mean_random_macs = torch.mean(random_macs_count_T, axis=0)
        # min_random_macs = torch.min(random_macs_count_T, axis=0)[0]
        # max_random_macs = torch.max(random_macs_count_T, axis=0)[0]
        # sns.lineplot(
        #     data=random_macs_df, 
        #     x="Threshold", 
        #     y="MACs", 
        #     estimator="mean", 
        #     errorbar="sd", 
        #     label="Random Mean Macs",
        #     color=random_agent_color,
        #     alpha=random_agent_alpha,
        #     linestyle="--",
        #     ax=ax2, # Pre-existing axes for the plot
        # )
        
        mean_mean_random_macs = torch.mean(mean_random_macs).item()
        yticks.append(mean_mean_random_macs)
        ylabels.append(f"{mean_mean_random_macs:.2f} (b)")

    # Set the yticks and labels
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)

    # Change the color of the label of just the extra number
    ytick_labels = ax2.get_yticklabels()
    for label in ytick_labels:
        if label.get_text() == f"{backbone_mmacs:.2f} (a)":
            label.set_color("grey")

        if label.get_text() == f"{mean_mean_random_macs:.2f} (b)":
            label.set_color("grey")

    # # if thresholds is not None:
    #     # ax1.xaxis.set_xticklabels(thresholds)

    # if rotate_labels:
    #     fig.set_xticklabels(fig.get_xticklabels(), rotation=20)

    # plt.legend()
    plt.savefig(f"{result_dir}/reward_vs_macs.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def extract_score_from_multiple_files(timestamp_list):

    msrn_rewards = None
    random_rewards = None
    backbone_rewards = None
    msrn_macs_list= None
    random_macs_list = None

    msrn_macs_mean_list = list()
    random_macs_mean_list = list()
    

    for _, timestamp in enumerate(timestamp_list):
        eval_results_dir = f"evaluation_results/{timestamp}"

        threshold_file = f"{eval_results_dir}/exit_threshold.json"
        threshold_list = load_json_as_list(threshold_file)

        score_file = f"{eval_results_dir}/rewards.json"
        scores = load_json_as_list(score_file)
        scores = torch.tensor(scores).squeeze()
        
        if msrn_rewards is None:
            msrn_rewards = scores[:, -1].unsqueeze(0)
            random_rewards = scores[:, -2].unsqueeze(0)
            backbone_rewards = scores[:, -3].unsqueeze(0)
        else:
            new_msrn_scores = scores[:, -1].unsqueeze(0)
            new_random_scores = scores[:, -2].unsqueeze(0)
            new_backbone_scores = scores[:, -3].unsqueeze(0)
            msrn_rewards = torch.cat((msrn_rewards, new_msrn_scores), dim=0)
            random_rewards = torch.cat((random_rewards, new_random_scores), dim=0)
            backbone_rewards = torch.cat((backbone_rewards, new_backbone_scores), dim=0)

        exit_dist_file = f"{eval_results_dir}/exit_points.json"
        exit_dists = load_json_as_list(exit_dist_file)
        exit_dists = torch.tensor(exit_dists).squeeze()
        
        msrn_macs_mean, random_macs_mean, _ = get_all_mean_macs(exit_dists)
        msrn_macs_all, random_macs_all, _ = get_all_macs(exit_dists)

        msrn_macs_mean_list.append(msrn_macs_mean.item())
        random_macs_mean_list.append(random_macs_mean.item())

        if msrn_macs_list is None:
            msrn_macs_list = msrn_macs_all.unsqueeze(0)
            random_macs_list = random_macs_all.unsqueeze(0)
        else:
            msrn_macs_list = torch.cat((msrn_macs_list, msrn_macs_all.unsqueeze(0)))
            random_macs_list = torch.cat((random_macs_list, random_macs_all.unsqueeze(0)))

    
    return msrn_rewards, random_rewards, backbone_rewards, msrn_macs_list, random_macs_list

def main():
    # timestamp_list = [
    #     "1700145488_34_comp_0_4",
    #     "1700145498_34_comp_0_45",
    #     "1700145603_34_comp_0_5",
    #     "1700145639_34_comp_0_55",
    #     "1700145831_34_comp_0_6",
    #     "1700145857_34_comp_0_65",
    #     "1700146049_34_comp_0_7",
    #     "1700146094_34_comp_0_75",
    #     "1700146302_34_comp_0_8",
    #     "1700146439_34_comp_0_85",
    #     "1700146547_34_comp_0_9",
    # ]
    # timestamp_list = [
    #     "1700246057_18_comp_exploding",
    #     "1700247231_18_comp_0_65", 
    # ]
    # timestamp_list = [
    #     "1700383356_34_comp_e2e",
    #     "1700145857_34_comp_0_65", 
    # ]
    # timestamp_list = [
    #     "1700247231_18_comp_0_65",
    #     "1700145857_34_comp_0_65", 
    # ]
    # timestamp_list = [
    #     "1700145857_34_comp_0_65",
    #     "1700306774_34_comp_linear_0_65", 
    # ]
    timestamp_list = [
        "1700145857_34_comp_0_65",
        "1700228586_34_no_comp_0_65", 
    ]


    msrn_rewards, random_rewards, backbone_rewards, msrn_macs_count, random_macs_count = extract_score_from_multiple_files(timestamp_list)
    
    tested_threshold = np.arange(40, 95, 5) / 100
    num_threshold_tested = len(tested_threshold)

    # plot_split_violin_plot(msrn_rewards, training_type="e2e", result_dir="./reports")
    # plot_split_violin_plot_18_vs_34(msrn_rewards, result_dir="./reports")

    # plot_split_violin_plot_pareto_vs_linear(msrn_rewards, result_dir="./reports")

    plot_split_violin_plot_comp_vs_no_comp(msrn_rewards, result_dir="./reports")

    exit()

    # timestamp = "1700141473_18_comp_correct_state" #"1700130463_18_comp_exploding" int(1699630482)
    for timestamp in timestamp_list:
        eval_results_dir = f"evaluation_results/{timestamp}"
        # eval_results_dir = f"results/{timestamp}"

        threshold_file = f"{eval_results_dir}/exit_threshold.json"
        threshold_list = load_json_as_list(threshold_file)
        # threshold_list = [0.9, 0.9, 0.9]

        score_file = f"{eval_results_dir}/rewards.json"
        # score_file = f"{eval_results_dir}/scores.json"
        scores = load_json_as_list(score_file)
        scores = torch.tensor(scores).squeeze()

        # plot_scores_from_nested_list(scores, labels=["train"], result_dir=eval_results_dir)
        # (
        #     msrn_mean_reward[idx],
        #     random_reward[idx],
        #     backbone_reward[idx],
        # ) = get_all_mean_scores(scores)

        exit_dist_file = f"{eval_results_dir}/exit_points.json"
        exit_dists = load_json_as_list(exit_dist_file)
        exit_dists = torch.tensor(exit_dists).squeeze()
        # msrn_macs_count[idx], random_macs_count[idx], _ = get_all_mean_macs(exit_dists)

        action_dist_file = f"{eval_results_dir}/action_dist.json"
        action_dist = load_json_as_list(action_dist_file)
        action_dist = torch.tensor(action_dist).squeeze()

        num_ee = (exit_dists.shape[-1]) - 1

        new_labels = create_dynamic_list(num_ee)

        # plot_action_distribution(action_dist, new_labels, result_dir=eval_results_dir)

        # plot_macs_for_agent(exit_dists, result_dir=eval_results_dir, labels=new_labels, exit_threshold=threshold_list, timestamp=timestamp)
        # plot_exit_distribution(exit_dists, result_dir=eval_results_dir, exit_threshold=threshold_list)
        # plot_exit_distribution(exit_dists, agent_type="random", result_dir=eval_results_dir, exit_threshold=threshold_list)

        # plot_reward_for_each_agent(
        #     scores,
        #     plot_type="violin",
        #     result_dir=eval_results_dir,
        #     labels=new_labels,
        #     exit_threshold=threshold_list,
        # )

    plot_reward_vs_macs(
            msrn_rewards,
            msrn_macs_count,
            random_macs_count=random_macs_count,
            random_reward=random_rewards,
            backbone_reward=backbone_rewards,
            thresholds=tested_threshold,
            result_dir="./reports",
        )

# timestamp_list = [
#         "1699968537_34_comp_0_4",
#         "1699968667_34_comp_0_45",
#         "1699968717_34_comp_0_5",
#         "1699968776_34_comp_0_55",
#         "1699969002_34_comp_0_6",
#         "1699983502_34_comp_0_65",
#         "1699983583_34_comp_0_7",
#         "1699995739_34_comp_0_75",
#         "1699995772_34_comp_0_8",
#         "1699995962_34_comp_0_85",
#         "1699999604_34_comp_0_9",
#     ]
#     tested_threshold = np.arange(40, 95, 5) / 100
#     num_threshold_tested = len(tested_threshold)
#     msrn_mean_reward = np.zeros((num_threshold_tested))
#     random_reward = np.zeros((num_threshold_tested))
#     backbone_reward = np.zeros((num_threshold_tested))
#     msrn_macs_count = np.zeros((num_threshold_tested))
#     random_macs_count = np.zeros((num_threshold_tested))

#     for idx, timestamp in enumerate(timestamp_list):
#         eval_results_dir = f"evaluation_results/{timestamp}"

#         threshold_file = f"{eval_results_dir}/exit_threshold.json"
#         threshold_list = load_json_as_list(threshold_file)

#         score_file = f"{eval_results_dir}/rewards.json"
#         scores = load_json_as_list(score_file)
#         scores = torch.tensor(scores).squeeze()
#         (
#             msrn_mean_reward[idx],
#             random_reward[idx],
#             backbone_reward[idx],
#         ) = get_all_mean_scores(scores)

#         exit_dist_file = f"{eval_results_dir}/exit_points.json"
#         exit_dists = load_json_as_list(exit_dist_file)
#         exit_dists = torch.tensor(exit_dists).squeeze()
#         msrn_macs_count[idx], random_macs_count[idx], _ = get_all_mean_macs(exit_dists)

#         action_dist_file = f"{eval_results_dir}/action_dist.json"
#         action_dist = load_json_as_list(action_dist_file)
#         action_dist = torch.tensor(action_dist).squeeze()


if __name__ == "__main__":
    main()
