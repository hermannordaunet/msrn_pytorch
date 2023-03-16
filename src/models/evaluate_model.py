# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

def main():
    model_path = "/Users/hermannordaunet/Documents/UiO/Master/Master-Codebase/models/model.pt"
    data_path = "/Users/hermannordaunet/Documents/UiO/Master/Master-Codebase/data"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print("[INFO] loading the KMNIST test dataset...")
    testData = KMNIST(root=data_path, train=False, download=True,
        transform=ToTensor())
    idxs = np.random.choice(range(0, len(testData)), size=(10,))
    testData = Subset(testData, idxs)
    # initialize the test data loader
    testDataLoader = DataLoader(testData, batch_size=1)
    # load the model and set it to evaluation mode
    model = torch.load(model_path, map_location=device).to(device)
    model.eval()

    with torch.no_grad():
	# loop over the test set
        for (image, label) in testDataLoader:
            # grab the original image and ground truth label
            origImage = image.numpy().squeeze(axis=(0, 1))
            gtLabel = testData.dataset.classes[label.numpy()[0]]
            # send the input to the device and make predictions on it
            image = image.to(device)
            pred, exit_idx, _, conf  = model(image)
            # find the class label index with the largest corresponding
            # probability
            idx = pred.argmax(axis=1).cpu().numpy()[0]
            predLabel = testData.dataset.classes[idx]

            # convert the image from grayscale to RGB (so we can draw on
            # it) and resize it (so we can more easily see it on our
            # screen)
            origImage = np.dstack([origImage] * 3)
            origImage = imutils.resize(origImage, width=128)
            # draw the predicted class label on it
            color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
            cv2.putText(origImage, gtLabel, (2, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
            # display the result in terminal and show the input image
            print("[INFO] ground truth label: {}, predicted label: {}, exit: {}, conf: {}".format(
                gtLabel, predLabel, exit_idx, conf))
            cv2.imshow("image", origImage)
            cv2.waitKey(0)



if __name__ == "__main__":
    main()
