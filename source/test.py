import os
import random
import torch
import cv2 as cv
import custom_dataset
from globals import *
import architectures.cnn
import architectures.inceptionet
import architectures.resnet
import architectures.inceptionv4
import architectures.incresnet
import custom_dataset
# Load the saved model
model = torch.load(path_cnn_file)
model.eval()  # Set the model to evaluation mode


# Select 20 random images from the directory

def main():
    training_data = []
    file_list = os.listdir(path_validation_set)
    random.shuffle(file_list)
    for filename in file_list:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(path_validation_set, filename)
            label = float(filename.split('_')[0])
            training_data.append((cv.imread(file_path), custom_dataset.class_labels_reassign(label)))
        if len(training_data) >= 100:
            break
    dataset = custom_dataset.CustomDataset(training_data)

    batch_size = 1
    shuffle = True
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print("loaded")
    error = 0
    count = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data


        outputs = model(inputs)
        predic = outputs.item()
        actual = labels.item()
        print(f"Actual: {actual}\tPredicted: {predic}")
        count += 1
        error += abs(predic -  actual)
           
    print(f"Mean error over {count} samples: {error / count}")

if __name__ == "__main__":
    main()