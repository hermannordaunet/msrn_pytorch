import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# import the necessary torch packages
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# import the necessary torchvision packages
from torchvision.datasets import KMNIST, CIFAR10, ImageNet
from torchvision.transforms import ToTensor

# import the necessary sklearn packages
from sklearn.metrics import classification_report

# import the necassary scipy packages
from scipy import stats

# Local import
from small_dqn import small_DQN
from small_dqn_ee import small_DQN_EE
from ee_cnn_residual import EE_CNN_Residual
from utils.loss_functions import loss_v1, loss_v2
from utils.data_utils import min_max_conf_from_dataset
from utils.print_utils import print_min_max_conf, print_cost_of_exits


def train(model, train_loader, optimizer, device: str()):
    losses, pred_losses, cost_losses = (
        list(),
        list(),
        list(),
    )
    conf_min_max = list()
    num_ee = len(model.exits)

    # setting model in train mode.
    model.train()
    # loop over the training set
    for data, target in train_loader:
        # send the input to the device
        data, target = data.to(device), target.to(device, dtype=torch.int64)

        # TODO: Find out when to use this
        optimizer.zero_grad()

        # perform a forward pass and calculate the losses
        if isinstance(model, small_DQN_EE):
            pred, conf, cost = model(data)
            cost.append(torch.tensor(1.0).to(device))
            conf_min_max.append(conf)

            cumulative_loss, pred_loss, cost_loss = loss_v2(
                pred, target, conf, cost, num_ee=num_ee
            )
        elif isinstance(model, EE_CNN_Residual):
            pred, conf, cost = model(data)
            cost.append(torch.tensor(1.0).to(device))
            conf_min_max.append(conf)

            cumulative_loss, pred_loss, cost_loss = loss_v2(
                pred, target, conf, cost, num_ee=num_ee
            )
        else:
            print("No training loop implemented for other model architectures")
            # TODO: Add exit code to this exit.
            exit()

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        losses.append(float(cumulative_loss))
        pred_losses.append(float(pred_loss))
        cost_losses.append(float(cost_loss))
        cumulative_loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions

        # TODO: Check if we need this
        # totalTrainLoss += loss

        # trainCorrect += (pred.argmax(1) == target).type(torch.float).sum().item()

    return losses, pred_losses, cost_losses, conf_min_max


def main():
    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64  # CRITICAL: exit block cant handle batch size > 1 in eval mode
    EPOCHS = 25

    # define the train and val splits
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # For numpy and torch random
    SEED = 1804

    # set the device we will be using to train the model
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] Device is: {device}")

    # load the KMNIST dataset
    print("[INFO] loading the KMNIST dataset...")
    trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
    testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

    # Load the sizes of dataset
    NUM_CLASSES = len(trainData.classes)
    _, IMG_WIDTH, IMG_HEIGHT = trainData.data.shape
    IN_CHANNELS = 1  # TODO: Make this more dynamic?

    # calculate the train/validation split
    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
    numValSamples = int(len(trainData) * VAL_SPLIT)
    (trainData, valData) = random_split(
        trainData,
        [numTrainSamples, numValSamples],
        generator=torch.Generator().manual_seed(SEED),
    )

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=TEST_BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=TEST_BATCH_SIZE)
    # calculate steps per epoch for training and validation set
    # trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // TEST_BATCH_SIZE

    # initialize the small DQN EE model
    # print("[INFO] initializing the small_DQN model...")
    # model = small_DQN_EE(
    #     input_shape=(IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH),
    #     num_classes=NUM_CLASSES,
    # ).to(device)

    print("[INFO] initializing the EE_CNN_Residual model...")
    model = EE_CNN_Residual(
        input_shape=(IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH),
        num_classes=NUM_CLASSES,
        num_ee=2,
        exit_threshold=0.99,
        repetitions=[2, 2],
        planes=[32, 64, 64],
        distribution="pareto",
    ).to(device)

    # initialize our optimizer
    optimizer = AdamW(model.parameters(), lr=INIT_LR)

    print_cost_of_exits(model)

    # # initialize a dictionary to store training history
    # H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    print("[INFO] training the network...")

    # Timing the training and validation loop
    startTime = time.time()
    print(f"[INFO] started training @ {time.ctime(startTime)}")

    for e in range(0, EPOCHS):
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))

        losses, pred_losses, cost_losses, batch_confs = train(
            model, trainDataLoader, optimizer, device
        )

        min_vals, max_vals = min_max_conf_from_dataset(batch_confs)
        print_min_max_conf(min_vals, max_vals)

        totalValLoss = 0
        valCorrect = 0

        exit_points = [0] * (len(model.exits) + 1)
        conf_min_max = list()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for data, target in valDataLoader:
                # send the input to the device
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                # make the predictions and calculate the validation loss
                pred, idx, cost, conf = model(data)
                conf_min_max.append([conf])
                loss = torch.nn.functional.nll_loss(pred, target) + 1.0 * cost
                exit_points[idx] += 1

                totalValLoss += loss
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == target).type(torch.float).sum().item()

        # min_vals, max_vals = min_max_conf_from_dataset(batch_confs)
        # print_min_max_conf(min_vals, max_vals)
        # trainCorrect = trainCorrect / len(trainDataLoader.dataset)

        avgValLoss = totalValLoss / valSteps
        valCorrect = valCorrect / len(valDataLoader.dataset)

        result = {
            "train_loss": round(np.mean(losses), 4),
            "train_loss_sem": round(stats.sem(losses), 2),
            "pred_loss": round(np.mean(pred_losses), 4),
            "pred_loss_sem": round(stats.sem(pred_losses), 2),
            "cost_loss": round(np.mean(cost_losses), 4),
            "cost_loss_sem": round(stats.sem(cost_losses), 2),
        }

        # print(f"\nResults:\n{result}\n")

        # print the model training and validation information
        print(
            "Train loss: {:.6f}, Prediction loss: {:.4f}, Cost Loss: {:.4f}".format(
                result["train_loss"], result["pred_loss"], result["cost_loss"]
            )
        )
        print(
            "Average val loss: {:.6f}, Val accuracy: {:.4f}\n, Exit points: {}".format(
                avgValLoss, valCorrect, exit_points
            )
        )

        # min_vals, max_vals = get_max_min_conf(conf_min_max)

        # print(f"\n[EVAL]: Min values at each exit: {min_vals}")
        # print(f"[EVAL]: Max values at each exit: {max_vals}\n")

    # finish measuring how long training took
    endTime = time.time()
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )
    # we can now evaluate the network on the test set
    print("[INFO] evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = list()
        # loop over the test set
        for data, _ in testDataLoader:
            # send the input to the device
            data = data.to(device)
            # make the predictions and add them to the list
            pred, idx, cost, conf = model(data)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    # generate a classification report
    print(
        classification_report(
            testData.targets.cpu().numpy(),
            np.array(preds),
            target_names=testData.classes,
        )
    )

    # # plot the training loss and accuracy
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(H["train_loss"], label="train_loss")
    # plt.plot(H["val_loss"], label="val_loss")
    # plt.plot(H["train_acc"], label="train_acc")
    # plt.plot(H["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy on Dataset")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig(r"./reports/figures/plot.png")
    # # serialize the model to disk
    torch.save(model, r"./models/loss_v1_KMNIST_model.pt")


if __name__ == "__main__":
    main()
