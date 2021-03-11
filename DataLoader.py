import torch
import torchvision
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.types as types

class HybridPipelineTrain(Pipeline):
    def __init__(self, batch_size, output_size, num_threads, device_id, images_directory):
        super(HybridPipelineTrain, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = images_directory, random_shuffle = True, initial_fill = 21)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmn = ops.CropMirrorNormalize(
                       device="gpu",
                       dtype=types.FLOAT,
                       mean=[0.5, 0.5, 0.5],
                       std=[0.5, 0.5, 0.5],
                       output_layout="HWC"
                   )
        self.coin = ops.random.CoinFlip(probability = 0.5)
        self.flip = ops.Flip(device = "gpu")
        self.rsz = ops.Resize(resize_x = output_size[0], resize_y = output_size[1], device = "gpu")

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rsz(images)
        images = self.flip(images, horizontal = self.coin())
        images = self.cmn(images)
        # images are on the GPU
        return (images, labels)

# Create the dataloader

def DataLoader(DATA_SIZE, batch_size, output_size, train_data_path, nvidiadali = False):
    if nvidiadali:
        pipe = HybridPipelineTrain(batch_size=batch_size, output_size = output_size, num_threads=2, device_id=0, images_directory=train_data_path)
        pipe.build()

        ITERATIONS_PER_EPOCH = DATA_SIZE // batch_size

        train_data_loader = DALIClassificationIterator([pipe], reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    else:
        img_transforms = torchvision.transforms.Compose([
	        torchvision.transforms.Resize(output_size),
        	torchvision.transforms.RandomHorizontalFlip(),
	        torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	    ])
        dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms)
        train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
    return train_data_loader