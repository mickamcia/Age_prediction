import os
import torch
import cv2 as cv
from globals import *
import architectures.cnn
import architectures.inceptionet
import architectures.resnet
import architectures.inceptionv4
import architectures.incresnet
import architectures.mobilenet
import architectures.xception
import architectures.vgg19
import architectures.resnext
import architectures.xception
import custom_dataset


def main():
    training_data = []
    for filename in os.listdir(path_training_set_augmented):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(path_training_set_augmented, filename)
            label = float(filename.split('_')[0])
            training_data.append((cv.imread(file_path), custom_dataset.class_labels_reassign(label)))

    dataset = custom_dataset.CustomDataset(training_data)

    print_every = 10
    num_epochs = 100
    batch_size = 16
    shuffle = True
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print("loaded")
    cnn = architectures.cnn.CNN() #this is my, small and generic CNN
    #cnn = architectures.resnet.ResNet152(3, 1) # ResNet50, ResNet101 or ResNet152, increasing in size
    #cnn = architectures.inceptionet.GoogLeNet(num_classes=1)
    #cnn = architectures.inceptionv4.Inceptionv4(3,1)
    #cnn = architectures.incresnet.Inception_ResNetv2(3,1)
    #cnn = architectures.mobilenet.MobileNetV2(3,1)
    #cnn = architectures.vgg19.VGG19(1)
    #cnn = architectures.resnext.resnext50()
    #cnn = architectures.xception.Xception(1)
    #cnn = torch.load(path_cnn_file) # use this to load current cnn instead of starting over
    if GPU_ENABLED:
        cnn.cuda()  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001, )
    #optimizer = torch.optim.Adagrad(cnn.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            labels = labels.view(-1, 1).float()
            optimizer.zero_grad()
            if GPU_ENABLED:
                labels = labels.cuda()
                inputs = inputs.cuda()
            outputs = cnn(inputs)
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

        if GPU_ENABLED:
            torch.save(cnn.cpu(), path_cnn_file + "_" + str(epoch))
            cnn.cuda()
            print("saved")
        else:
            torch.save(cnn, path_cnn_file + "_" + str(epoch))
            print("saved")

if __name__ == "__main__":
    main()