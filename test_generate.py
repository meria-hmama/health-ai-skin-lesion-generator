import torch                                  
from torchvision.utils import save_image       # saves a tensor as a picture file
from model import Generator, latent_dim        

netG = Generator()                             # build an empty generator with the same layers
netG.load_state_dict(torch.load("generator.pth", map_location="cpu"))  # fill it with my  trained weights
netG.eval()                                    # switch to prediction mode 

noise = torch.randn(1, latent_dim, 1, 1)       # one random list of 100 numbers
with torch.no_grad():                          # no training here
    img = netG(noise)                          # generate the image

save_image((img + 1) / 2, "test.png")          
print("saved test.png")                        