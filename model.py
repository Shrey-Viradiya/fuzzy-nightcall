import torch
import torchvision.utils as vutils
from generator import Generator
from discriminator import Discriminator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class Model:
    def __init__(self, latent_vector_size, generator_feature_map_size, discriminator_feature_map_size, target_channels):
        # creating models
        self.latent_vector_size = latent_vector_size
        self.generator_feature_map_size = generator_feature_map_size
        self.discriminator_feature_map_size = discriminator_feature_map_size
        self.target_channels = target_channels

        self.netG = Generator(latent_vector_size=latent_vector_size, 
                        generator_feature_map_size=generator_feature_map_size, 
                        output_channel=target_channels)

        self.netD = Discriminator(discriminator_feature_map_size=discriminator_feature_map_size,
                            input_channel=target_channels)

        # Establish convention for real and fake labels during training
        

    def train(self, train_data, optimizerG, optimizerD, fixed_noise, epochs = 5, criterion = torch.nn.BCELoss(), device = "cuda", dali = False):
        # Lists to keep track of progress
        G_losses = []
        D_losses = []
        iters = 0
        real_label = 1.0
        fake_label = 0.0
        checkpoint = len(train_data)//4

        for epoch in range(epochs):
            # For each batch in the dataloader
            for i, data in enumerate(train_data):
                start = time.time()
                if dali:
                    d = data[0]
                    train_images, train_labels = d["data"], d["label"]
                    train_images = train_images.to(device)
                    train_images = train_images.permute(0,3,1,2)
                else:
                    train_images, train_labels = data
                    train_images = train_images.to(device)
                self.netD.to(device)
                self.netG.to(device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                b_size = train_images.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = self.netD(train_images).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.latent_vector_size, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % checkpoint == 0:
                    print(f"[{epoch}/{epochs}] [{i}/{len(train_data)}] \tLoss_D: {errD.item()} \tLoss_G: {errG.item()} \tD(x): {D_x} \tD(G(z)): {D_G_z1}/{D_G_z2}\ttime:{time.time()-start}")

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % checkpoint == 0) or ((epoch == epochs-1) and (i == len(train_data)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    fig = plt.figure(figsize=(8,8))
                    plt.axis("off")
                    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
                    plt.savefig(f"./progress/{epoch}_{iters}.png")
                    torch.save(self.netD, "./models/{epoch}_{iters}.pkl")
                    np.save("./models/G_losses.npy", np.array(G_losses))
                    np.save("./models/D_losses.npy", np.array(D_losses))
                    plt.close(fig)

                iters += 1
