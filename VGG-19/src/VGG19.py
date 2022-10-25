import torch.nn as nn

from torch import flatten

class VGG19(nn.Module):
    """
    VGG19 architecture, variants of VGG (Simonyan & Zisserman, 2014) addapted to CIFAR10
    """

    def __init__(self):
        super().__init__()

        self.features_extractor_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.features_extractor_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.features_extractor_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.features_extractor_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.features_extractor_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=10),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data[m.weight.data==0] = 0.005

                nn.init.zeros_(m.bias.data)

        self.apply(weights_init)


    def forward(self, input_tensor):

        output = self.features_extractor_block_1(input_tensor)
        output = self.features_extractor_block_2(output)
        output = self.features_extractor_block_3(output)
        output = self.features_extractor_block_4(output)
        output = self.features_extractor_block_5(output)

        output = flatten(
            input=output,
            start_dim=1
        )

        return self.classifier(output)


if __name__ == "__main__":
    import pytorch_model_summary as pms

    from torch import randn as torch_randn

    model = VGG19()

    print("All Weights")
    print(pms.summary(model, torch_randn((1, 3, 32, 32))))

    print("\Linear Weights")
    print(pms.summary(
        model.classifier,
        torch_randn((1, 512))
    ))


    print("\nConv Weights")
    print(pms.summary(
        nn.Sequential(
            model.features_extractor_block_1,
            model.features_extractor_block_2,
            model.features_extractor_block_3,
            model.features_extractor_block_4,
            model.features_extractor_block_5,
        ),
        torch_randn((1, 3, 32, 32))
    ))
