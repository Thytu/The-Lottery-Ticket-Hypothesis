import torch.nn as nn

from torch import flatten

class LeNet(nn.Module):
    # TODO: doc

    def __init__(self):
        super(LeNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=300),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=100, out_features=10),
        )


    def forward(self, input_tensor):

        input_tensor = flatten(
            input=input_tensor,
            start_dim=1
        )

        return self.classifier(input_tensor)
