import torch.nn as nn

from torch import flatten

class Conv2(nn.Module):
    """
    Conv2 architecture, variants of VGG (Simonyan & Zisserman, 2014)
    """

    def __init__(self):
        super().__init__()

        self.features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16_384, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=10),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        self.apply(weights_init)


    def forward(self, input_tensor):

        output = self.features_extractor(input_tensor)


        output = flatten(
            input=output,
            start_dim=1
        )

        return self.classifier(output)


if __name__ == "__main__":
    import pytorch_model_summary as pms

    from torch import randn as torch_randn

    model = Conv2()

    print("All Weights")
    print(pms.summary(model, torch_randn((1, 3, 32, 32))))

    print("\nConv Weights")
    print(pms.summary(model.features_extractor, torch_randn((1, 3, 32, 32))))
