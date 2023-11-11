import json
import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    dist: list(),
    plot_type="box",
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
):
    df = pd.DataFrame(dist)

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
        plot = sns.violinplot(x="Agent", y="Reward", data=df_melted)

    plt.title("Rewards for each agent type")
    plt.xlabel("Agents")
    plt.ylabel("Reward")

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/rewards_{plot_type}.pdf", format="pdf", bbox_inches="tight"
    )


def plot_action_distribution(
    score: list(),
    plot_type="box",
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
):
    df = pd.DataFrame(
        [episode[-1] for episode in data if isinstance(episode, list) and episode]
    )

    # df_melted = df.melt(
    #     var_name="Agent",
    #     value_name="Actions",
    # )

    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    sns.despine()

    plot = sns.barplot(
        x="Action",
        y="Amount",
        data=df_melted,
    )

    plt.title("Boxplot of agent scores in an environment")
    plt.xlabel("Agents")
    plt.ylabel("Actions")

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/action_dist_{plot_type}.pdf", format="pdf", bbox_inches="tight"
    )


def plot_exit_distribution(
    exit_dist: list(),
    agent_type="msrn",
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
):
    if agent_type == "msrn":
        last_agent_data = exit_dist[:, -1, :]
    elif agent_type == "random":
        last_agent_data = exit_dist[:, -2, :]
    else:
        print("No supported agent type provided. Plotting MSRN exit distribution.")
    number_of_episodes = exit_dist.shape[0]
    number_of_exits = exit_dist.shape[-1]
    number_of_agents = exit_dist.shape[1]

    exit_numbers = np.tile(
        np.arange(number_of_exits), (number_of_episodes, 1)
    )  # Repeating exit numbers for each episode
    episode_numbers = np.repeat(
        np.arange(1, number_of_episodes+1), number_of_exits
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
    plot = sns.barplot(data=df, x="Exit", y="Count", estimator=np.mean, errorbar="sd")
    plt.title("Average Exit Count for the Last Agent with Standard Deviation")
    sns.despine()

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/exit_dist_{agent_type}.pdf", format="pdf", bbox_inches="tight"
    )


def get_macs_for_agent(exit_dist, exit_macs):

    number_of_steps = torch.sum(exit_dist[0,0,0])
    result = torch.einsum('ijk,k->ij', exit_dist.type(torch.float32), exit_macs) / number_of_steps

    return result

def plot_macs_for_agent(
    exit_dist: list(),
    labels=None,
    env_name="",
    result_dir="./",
    rotate_labels=True,
):
    
    num_exits = exit_dist.shape[-1]
    
    if num_exits == 4:
        cost_list = torch.tensor([21.22, 34.36, 42.63, 86.84])
    elif num_exits == 6:
        cost_list= torch.tensor([34.38, 64.83, 86.09, 107.35, 117.98, 170.45])
    else:
        print("No exit cost_list for this amount of exits.")
        exit()

    tensor = get_macs_for_agent(exit_dist, cost_list)

    # Now create the barplot using Seaborn
    plt.figure(figsize=set_size())
    sns.set_theme(style="darkgrid")
    plot = sns.barplot(data=tensor, estimator=np.mean, errorbar="sd", capsize=.1)
    plt.title("MACs for each agent. With Standard Deviation for Random and MSRN")
    sns.despine()

    if labels is not None:
        plot.set_xticklabels(labels)

    if rotate_labels:
        plot.set_xticklabels(plot.get_xticklabels(), rotation=20)

    plt.savefig(
        f"{result_dir}/macs.pdf", format="pdf", bbox_inches="tight"
    )


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
    scores: list(), labels=None, env_name="", result_dir="./"
):
    scores = np.array(scores)

    if labels:
        if len(labels) == 1:
            labels = labels[-1]

        # Calculate mean, min, and max values for each time point
    mean_values = np.mean(scores, axis=1)
    min_values = np.min(scores, axis=1)
    max_values = np.max(scores, axis=1)

    # Create x-axis values (time points)
    x = np.arange(1, scores.shape[0] + 1)

    # Set Seaborn style
    sns.set(style="darkgrid")

    # Create the line plot with error bars
    plt.figure(figsize=set_size())
    sns.lineplot(x=x, y=mean_values, label=labels)
    plt.fill_between(x, min_values, max_values, alpha=0.25)
    sns.despine()
    plt.title(f"{env_name} reward")
    plt.ylabel("Reward")
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


# save loss plot
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


# Eval loss and score


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

def main():
    timestamp = 1699645352
    eval_results_dir = f"evaluation_results/{timestamp}"

    score_file = f"{eval_results_dir}/rewards.json"
    scores = load_json_as_list(score_file)
    scores = [inner_list[0] for inner_list in scores]

    exit_dist_file = f"{eval_results_dir}/exit_points.json"
    exit_dists = load_json_as_list(exit_dist_file)
    exit_dists = torch.tensor(exit_dists).squeeze()

    action_dist_file = f"{eval_results_dir}/action_dist.json"
    action_dist = load_json_as_list(action_dist_file)
    action_dist = torch.tensor(action_dist).squeeze()

    num_ee = (exit_dists.shape[-1]) - 1

    new_labels = create_dynamic_list(num_ee)

    plot_macs_for_agent(exit_dists, result_dir=eval_results_dir, labels=new_labels)
    plot_exit_distribution(exit_dists, result_dir=eval_results_dir)
    # plot_exit_distribution(exit_dists, agent_type="random", result_dir=eval_results_dir)


    plot_reward_for_each_agent(
        scores,
        plot_type="violin",
        result_dir=eval_results_dir,
        labels=new_labels,
    )


if __name__ == "__main__":
    main()
