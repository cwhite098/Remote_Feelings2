import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import load_data, print_summary


class CustomImageDataset(Dataset):
    '''https://pytorch.org/tutorials/beginner/basics/data_tutorial.html'''
    def __init__(self, df, images, finger_name, device, transform=None, target_transform=None):

        self.images = images
        self.img_labels = df[['Image_Name','fz']]
        self.transform = transform
        self.target_transform = target_transform
        self.finger_name = finger_name
        self.device = device

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.as_tensor(image.reshape((1, image.shape[0], image.shape[1])), dtype = torch.float32, device=self.device)
        label = torch.as_tensor(self.img_labels.iloc[idx, 1], dtype = torch.float32, device=self.device)
        if self.transform:
            image = self.transform(image) # converts to tensor and normalises
        if self.target_transform:
            label = self.target_transform(label)
        if self.device=='cuda':
            return image, label
        else:
            return image, label


class Net(nn.Module):
    def __init__(self, finger_name, device):
        super().__init__()
        self.finger_name = finger_name

        self.convLayer1 = nn.Conv2d(1, 32, 3)
        self.convLayer2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(3, 1)
        self.hidden = nn.Linear(563200, 16)
        self.outLayer = nn.Linear(16,1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])

        self.losses = []
        self.device=device

    def forward(self, x):

        x = self.pool(F.relu(self.convLayer1(x))) # input convolution
        for i in range(4):
            x = self.pool(F.relu(self.convLayer2(x))) # subsequent convolutions

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.hidden(x))
        x = self.outLayer(x)
        return x

    def train(self, X_train, y_train, batch_size, epochs):

        trainset = CustomImageDataset(y_train, X_train, self.finger_name, self.device, transform=self.transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                labels = torch.reshape(labels, [batch_size,1])

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')
            self.losses.append(running_loss) # save the loss for plotting
            running_loss = 0.0

        print('Finished Training')






def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check if gpu available
    print(device)

    batch_size = 16 # from paper
    finger_name = 'Middle'

    print('Loading Data...')
    df, t1, t2, t3, blob_locs = load_data(finger_name)
    print(finger_name+':')
    print_summary(df)

    images = t1[1:] # remove default image
    X_train, X_test, y_train, y_test = train_test_split(images, df, test_size=0.2, random_state=42)

    net = Net(finger_name, device) # Init CNN
    net.to(device)
    net.train(X_train, y_train, batch_size, epochs=2) # train the CNN


if __name__ =='__main__':
    main()