import torch.nn as nn

# torch.nn.Conv2d(
#     in_channels, 
#     out_channels, 
#     kernel_size, 
#     stride=1, 
#     padding=0, 
#     dilation=1, 
#     groups=1, 
#     bias=True, 
#     padding_mode='zeros'
# )

# Define model
class ConvNet(nn.Module):
    # CNN층 정의
    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, 2, 1) # (input_channels, output_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.conv3 = nn.Conv2d(192, 384, 3, 2, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(384, 5)
        self.act = nn.ReLU()
       
    # CNN층 결합
    def forward(self, x): # conv2d_1 -> relu -> conv2d_2 -> relu -> conv2d_3 -> relu -> conv2d_4 -> relu -> mean -> dropout -> fully connected layer
        print(x.size())  # torch.Size([batch_size, channel, width, height])

        x = self.act(self.conv1(x))
        print(x.size())  # torch.Size([batch_size, channel, width, height])

        x = self.act(self.conv2(x))
        print(x.size())  # torch.Size([batch_size, channel, width, height])

        x = self.act(self.conv3(x))
        print(x.size())  # torch.Size([batch_size, channel, width, height])

        x = self.act(self.conv4(x))
        print(x.size())  # torch.Size([batch_size, channel, width, height])

        x = x.mean([-1, -2])
        print(x.size())  # torch.Size([batch_size, flatten])
        
        x = self.drop(x)
        print(x.size())  # torch.Size([batch_size, flatten])

        x = self.fc(x)
        print(x.size())  # torch.Size([batch_size, class])
        return x