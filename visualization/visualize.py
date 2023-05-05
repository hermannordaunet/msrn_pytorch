import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

# TODO: Add some plotting functions into this.
# Training loss and score


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
                print("Need provide enough labels to get legend on the plot")
                labels = None

    df = pd.DataFrame(scores.transpose(), columns=labels, index=x_values)

    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df)
    sns.despine()
    plt.title(f"{env_name} agent reward")
    plt.ylabel("Reward")
    plt.xlabel("Episode #")

    plt.xticks(x_values)

    if labels:
        plt.legend()

    plt.savefig(f"{result_dir}/{env_name}_scores.png")


# save loss plot
def plot_loss_from_list(losses: list(), labels=None, env_name="", result_dir="./"):
    losses = np.array(losses).squeeze()

    if losses.ndim > 1:
        x_values = np.arange(1, losses[-1].size + 1)
    else:
        x_values = np.arange(1, losses.size + 1)

    if labels:
        if losses.ndim > 1:
            if losses.shape[0] > len(labels):
                print("Need provide enough labels to get legend on the plot")
                labels = None

    df = pd.DataFrame(losses.transpose(), columns=labels, index=x_values)

    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df)
    sns.despine()
    plt.title(f"{env_name} loss")
    plt.ylabel("Loss")
    plt.xlabel("Episode #")

    plt.xticks(x_values)

    if labels:
        plt.legend()

    plt.savefig(f"{result_dir}/{env_name}_losses.png")


# Eval loss and score

# Grid images.