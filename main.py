import io                                           #building a file in memory, not on disk
import torch                                        #PyTorch
from fastapi import FastAPI, Response               #lets send raw image bytes
from torchvision.utils import save_image            #turning a tensor into a picture
from model import Generator, latent_dim             #my network and the number 100

app = FastAPI()                                     #the app object uvicorn runs

netG = Generator()                                  #building the generator once, when the server starts
netG.load_state_dict(torch.load("generator.pth", map_location="cpu"))  # load the trained weights
netG.eval()                                         #prediction mode

@app.get("/")                                       #home address
def home():
    return {"message": "Health-AI generator is running"}

@app.get("/generate")                               #new address: /generate
def generate():
    noise = torch.randn(1, latent_dim, 1, 1)        #a fresh random list of 100 numbers
    with torch.no_grad():                           #no training
        img = netG(noise)                           #generating the image
    buf = io.BytesIO()                              #empty in-memory file
    save_image((img + 1) / 2, buf, format="png")    #saving the image to that file(PNG format)
    return Response(content=buf.getvalue(), media_type="image/png")  #sending the image back