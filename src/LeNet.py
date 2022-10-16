import torch.nn as nn

from torch import flatten

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, input_tensor):

        input_tensor = self.feature_extractor(input_tensor)

        input_tensor = flatten(input=input_tensor, start_dim=1)

        return self.classifier(input_tensor)
