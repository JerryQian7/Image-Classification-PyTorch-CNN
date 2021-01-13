import torch.nn as nn


# Custom CNN model
class custom_Net(nn.Module):

    def __init__(self, classes):
        super(custom_Net, self).__init__()
        #new layer
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        #new layer
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.b5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        #new layer
        self.b6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        #new layer
#         self.b7 = nn.Sequential(
#             nn.Conv2d(512, 512, 3, stride = 1, padding = 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        #new layers
        out3 = self.b6(self.b5(out2))
#         out4 = self.b7(out3)
        out_avg = self.avg_pool(out4)

        out_flat = out_avg.view(-1, 512)
        out5 = self.fc2(self.fc1(out_flat))

        return out5
        