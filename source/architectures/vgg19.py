import torch.nn as nn
import torch.nn.functional as F

class VGG19(nn.Module):
  def __init__(self, classes):
    super(VGG19 , self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv2 = nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.conv3 = nn.Conv2d(in_channels=128 , out_channels=128 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv4 = nn.Conv2d(in_channels=128 , out_channels=256 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv5 = nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.conv6 = nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv7 = nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.maxPool = nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))

    self.fc1 = nn.Linear(in_features=18432 , out_features=4096)
    self.fc2 = nn.Linear(in_features=4096 , out_features=4096)
    self.fc3 = nn.Linear(in_features=4096 , out_features=1000)
    self.fc4 = nn.Linear(in_features=1000 , out_features=classes)

  def forward(self,x):
    # 2 Conv Layers with 64 kernels of size 3*3  
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 2 Conv Layers with 128 kernels of size 3*3
    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 2 Conv Layers with 256 kernels of size 3*3
    x = self.conv5(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = F.relu(x)    
    x = self.conv6(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 4 Conv Layers with 512 kernels of size 3*3
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 4 Conv Layers with 512 kernels of size 3*3
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)

    x = x.reshape(x.shape[0] , -1)

    #Fully Connected Layer With 4096 Units  
    x = self.fc1(x)
    x = F.relu(x)

    #Fully Connected Layer With 4096 Units
    x = self.fc2(x)
    x = F.relu(x)

    #Fully Connected Layer With 1000 Units
    x = self.fc3(x)
    x = self.fc4(x)
    return x