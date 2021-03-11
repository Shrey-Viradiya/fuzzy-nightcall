import torch

class Discriminator(torch.nn.Module):
    def __init__(self, discriminator_feature_map_size = 64, input_channel=3):
        super(Discriminator, self).__init__()
        
        self.main = torch.nn.Sequential(
            # input is (input_channel) x 128 x 128
            torch.nn.Conv2d(input_channel, discriminator_feature_map_size, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_feature_map_size) x 64 x 64
            torch.nn.Conv2d(discriminator_feature_map_size, discriminator_feature_map_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(discriminator_feature_map_size * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_feature_map_size*2) x 32 x 32
            torch.nn.Conv2d(discriminator_feature_map_size * 2, discriminator_feature_map_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(discriminator_feature_map_size * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_feature_map_size*4) x 16 x 16
            torch.nn.Conv2d(discriminator_feature_map_size * 4, discriminator_feature_map_size * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(discriminator_feature_map_size * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_feature_map_size*8) x 8 x 8
            torch.nn.Conv2d(discriminator_feature_map_size * 8, discriminator_feature_map_size * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(discriminator_feature_map_size * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),            
            # state size. (discriminator_feature_map_size*16) x 4 x 4
            torch.nn.Conv2d(discriminator_feature_map_size * 16, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
