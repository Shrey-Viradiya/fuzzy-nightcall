import torch

class Generator(torch.nn.Module):
	def __init__(self, latent_vector_size = 100, generator_feature_map_size = 64, output_channel=3):
		super(Generator, self).__init__()
		self.main = torch.nn.Sequential(
			# input is Z, going into a convolution 
			torch.nn.ConvTranspose2d(latent_vector_size, generator_feature_map_size * 8, 4, 1, 0, bias=False),
			torch.nn.BatchNorm2d(generator_feature_map_size * 8),
			torch.nn.ReLU(True),
			# state size. (generator_feature_map_size x 8) x 4 x 4
			torch.nn.ConvTranspose2d(generator_feature_map_size * 8, generator_feature_map_size * 4, 4, 2, 1, bias=False),
			torch.nn.BatchNorm2d(generator_feature_map_size * 4),
			torch.nn.ReLU(True),
			# state size. (generator_feature_map_size x 4) x 8 x 8
			torch.nn.ConvTranspose2d(generator_feature_map_size * 4, generator_feature_map_size * 2, 4, 2, 1, bias=False),
			torch.nn.BatchNorm2d(generator_feature_map_size * 2),
			torch.nn.ReLU(True),
			# state size. (generator_feature_map_size x 2) x 16 x 16
			torch.nn.ConvTranspose2d(generator_feature_map_size * 2, generator_feature_map_size, 4, 2, 1, bias=False),
			torch.nn.BatchNorm2d(generator_feature_map_size),
			torch.nn.ReLU(True),
			# state size. (generator_feature_map_size) x 32 x 32
			torch.nn.ConvTranspose2d(generator_feature_map_size, generator_feature_map_size, 4, 2, 1, bias=False),
			torch.nn.BatchNorm2d(generator_feature_map_size),
			torch.nn.ReLU(True),
			# state size. (generator_feature_map_size) x 64 x 64
			torch.nn.ConvTranspose2d( generator_feature_map_size, output_channel, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (output_channel) x 128 x 128
		)

	def forward(self, input):
		return self.main(input)
