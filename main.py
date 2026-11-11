import io                                           #building a file in memory, not on disk
import torch 
from fastapi.middleware.cors import CORSMiddleware                                       #PyTorch
from fastapi import FastAPI, Response               #lets send raw image bytes
from torchvision.utils import save_image            #turning a tensor into a picture
from model import Generator, latent_dim             #my network and the number 100

app = FastAPI()      
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])                                #the app object uvicorn runs

netG = Generator()                                  #building the generator once, when the server starts
netG.load_state_dict(torch.load("generator.pth", map_location="cpu"))  # load the trained weights
netG.eval()   
g = torch.Generator().manual_seed(1)                #makes numbers same everytime
z1 = torch.randn(1, latent_dim, 1, 1, generator=g)  #creates first fixed list of 100 numbers ( start point )
z2 = torch.randn(1, latent_dim, 1, 1, generator=g)  #creates second fixed list of 100 numbers ( end  point )

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
@app.get("/interpolate")                            #creates address
def interpolate(alpha: float = 0.5):                #from 0 to 1
    z = (1 - alpha) * z1 + alpha * z2               #blends two lists
    with torch.no_grad():                           #generate image without training 
        img = netG(z)                               #generate image without training 
    buf = io.BytesIO()                              #turn it into png
    save_image((img + 1) / 2, buf, format="png")    #send it back as in /generate 
    return Response(content=buf.getvalue(), media_type="image/png")