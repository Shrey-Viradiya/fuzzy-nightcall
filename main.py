import torch
import torchvision
from model import Model
import argparse
from DataLoader import DataLoader

DESCRIPTION = """
Fuzzy NightCall template repository
"""

USAGE = """python3 main.py --learning_rate 0.00001
Run Fuzzy NightCall Model on your machine.
For example, you can pass the following parameters:
python3 main.py --learning_rate 0.00001 --batch_size 32 --epoch 25 --data_dir "./data/img_align_celeba/img_align_celeba/"

"""

parser = argparse.ArgumentParser(description=DESCRIPTION, usage=USAGE)

parser.add_argument(
    "--learning_rate",
    metavar="learning_rate",
    type=float,
    action="store",
    help="Learning Rate for training",
    default=0.00001,
)
parser.add_argument(
    "--batch_size",
    metavar="batch_size",
    type=int,
    action="store",
    help="Batch_size",
    default=16,
)
parser.add_argument(
    "--epoch", metavar="epoch", type=int, action="store", help="Epoch", default=15
)
parser.add_argument(
    "--image_size", metavar="image_size", type=int, action="store", help="Output Image size", default=64
)
parser.add_argument(
    "--data_dir",
    metavar="data_dir",
    type=str,
    action="store",
    help="Data Directory Path",
    required=True,
)

parser.add_argument(
    "--nvidiadali",
    action="store_true",
    help="Option to use when want to use nvidiadali...need to install it before running",
)

if __name__ == "__main__":
	print("Fuzzy NightCall")
	kwargs = vars(parser.parse_args())

	if torch.cuda.is_available():
		print("Using GPU")
		device = torch.device("cuda")
	else:
		print("Using CPU")
		device = torch.device("cpu")

	# Create batch of latent vectors that we will use to visualize
	#  the progression of the generator
	generator_feature_map_size = 64
	discriminator_feature_map_size = 64
	target_channels = 3
	latent_vector_size = 100
	fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)
	output_size = [128, 128]

	dataloader = DataLoader(70000, kwargs['batch_size'], output_size, kwargs['data_dir'], kwargs['nvidiadali'])

	# Beta1 hyperparam for Adam optimizers
	beta1 = 0.5

	model = Model(latent_vector_size, generator_feature_map_size, discriminator_feature_map_size, target_channels)

	# Setup Adam optimizers for both G and D
	optimizerD = torch.optim.Adam(model.netD.parameters(), lr=kwargs['learning_rate'], betas=(beta1, 0.999))
	optimizerG = torch.optim.Adam(model.netG.parameters(), lr=kwargs['learning_rate'], betas=(beta1, 0.999))

	model.train(dataloader, optimizerG, optimizerD, fixed_noise, epochs=kwargs['epoch'], dali=kwargs['nvidiadali'])
