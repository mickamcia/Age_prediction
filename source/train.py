import os
import torch
import cv2 as cv
from globals import *
import architectures.cnn
import architectures.inceptionet
import architectures.resnet
import custom_dataset


def main():
    training_data = []
    for filename in os.listdir(path_training_set):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(path_training_set, filename)
            label = float(filename.split('_')[0])
            training_data.append((cv.imread(file_path), custom_dataset.class_labels_reassign(label)))

    dataset = custom_dataset.CustomDataset(training_data)

    print_every = 10
    num_epochs = 100
    batch_size = 16
    shuffle = True
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print("loaded")
    #cnn = definition.CNN() #this is my, small and generic CNN
    #cnn = resnet.ResNet152(3, 1) # ResNet50, ResNet101 or ResNet152, increasing in size
    #cnn = architectures.inceptionet.GoogLeNet(num_classes=1)
    cnn = torch.load(path_cnn_file) # use this to load current cnn instead of starting over
    cnn.cuda()  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            labels = labels.view(-1, 1).float().cuda()
            optimizer.zero_grad()
            outputs = cnn(inputs.cuda())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_every == print_every - 1:
                # print(labels)
                # print(outputs)
                # print(loss)
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {(running_loss / print_every) ** 0.5}')
                running_loss = 0.0

        torch.save(cnn.cpu(), path_cnn_file)
        cnn.cuda()
        print("saved")

if __name__ == "__main__":
    main()