import torch
from torch import nn
from hybrid_attention_machine import hybrid_attention_machine
class NormalCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(NormalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, 1),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            
       )
   
        self.attention=hybrid_attention_machine(32)
        self.BRXF = nn.Sequential(
            nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
           nn.Flatten())
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x=self.attention(x)
        x=self.BRXF(x)
        y = self.fc(x)
        return y


class AlexNet(nn.Module):
    """
    Implementation of AlexNet, from paper
    "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
    See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """

    def __init__(self, input_channels=3, num_classes=1000):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1)  # (b x 256 x 13 x 13)
           
            
        )
        self.attention=hybrid_attention_machine(256)
        #classifier is just a name for linear layers
        self.BRXF = nn.Sequential(
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.Flatten())
        

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)

        x=self.attention(x)
        x=self.BRXF(x)
        return self.fc(x)


if __name__ == '__main__':
    cnn = AlexNet(1)
    cnn.fc = nn.Identity()
    print(cnn(torch.randn(128, 1, 128, 128)).shape)
