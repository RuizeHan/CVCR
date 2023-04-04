import torch.nn as nn
import torch
import config

C = config.Config()


class S2CNet(nn.Module):
    def __init__(self):
        super(S2CNet, self).__init__()
        # self.vgg = vgg16()
        # models.vgg11()
        input_channel = 1 if C.INPUT_TYPE == 'map' else 3
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, (3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3),
        )
        self.FC_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # 529, 941

        x = x.type(torch.cuda.FloatTensor)
        x = x.permute(0, 3, 1, 2)
        x = self.classifier(x)
        xshape = x.shape
        x = torch.flatten(x, start_dim=1)

        x = self.FC_layers(x)

        return x