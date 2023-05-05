import numpy as np
import matplotlib.pyplot as plt

# TODO: Add some plotting functions into this.
# Training loss and score


# save scores plot
def plot_scores_from_list(scores: list(), labels=None, env_name="", result_dir="./"):
    scores = np.array(scores).squeeze()
    if scores.ndim > 1:
        x_values = np.arange(len(scores[-1]))
    else:
        x_values = np.arange(len(scores))

    if labels:
        if scores.ndim > 1:
            if len(scores) > len(labels):
                print("Need provide enough labels to get legend on the plot")
                labels = None

        else:
            labels = labels[0]

    plt.figure()
    plt.plot(x_values, scores.transpose(), label=labels)
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
        x_values = np.arange(len(losses[-1]))
    else:
        x_values = np.arange(len(losses))

    if labels:
        if losses.ndim > 1:
            if len(losses) > len(labels):
                print("Need provide enough labels to get legend on the plot")
                labels = None
        else:
            labels = labels[0]

    plt.figure()
    plt.plot(x_values, losses.transpose(), label=labels)
    plt.title(f"{env_name} loss")
    plt.ylabel("Loss")
    plt.xlabel("Episode #")

    plt.xticks(x_values)

    if labels:
        plt.legend()

    plt.savefig(f"{result_dir}/{env_name}_losses.png")


# Eval loss and score

# Grid images.