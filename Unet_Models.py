import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, is_res:bool = False)->None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channel = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res
        
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.GELU() # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.GELU() # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channel:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut =  nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(device)
                out = shortcut(x) + x2

            # Normalize output tensor
            return out / 1.414
        
        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)
    

class ToVec(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            nn.AvgPool2d((4)),
            nn.GELU()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
    
class InitializeUpSampling(nn.Module):
    def __init__(self, in_channels,out_channels,height):
        super().__init__()

        self.h = height

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.h//4, stride=self.h//4), # up-sample 
            nn.GroupNorm(8, out_channels), # normalize
            nn.ReLU()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

        
    
    
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x,skip), dim=1)

        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)

        return x
    

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()

        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''

        self.input_dim = input_dim

        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim) # making it 1 column vector
        # apply the model layers to the flattened tensor
        return self.model(x)


class InitializeOut(nn.Module):
    def __init__(self, in_channels, out_channels, IN_Channel):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, out_channels), # normalize
            nn.ReLU(),
            nn.Conv2d(out_channels, IN_Channel, kernel_size=3, stride=1, padding=1) # map to same number of channels as input

        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

        
