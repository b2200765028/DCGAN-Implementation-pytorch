from matplotlib.pyplot import title
import torch 
import torchvision
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator,Discriminator

NUM_EPOCHS = 20
LR = 0.0005
NOISE_DIM = 256
CHANNEL_IMG = 1
BATCH_SIZE = 64
IMG_SIZE = 64

FEATURES_D = 16
FEATURES_G = 16

transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])

dataset = datasets.MNIST(root="/dataset/",train=True,download=True,transform=transforms)
dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(NOISE_DIM,CHANNEL_IMG,FEATURES_G).to(device)
disc = Discriminator(CHANNEL_IMG,FEATURES_D).to(device)


optim_disc = optim.Adam(lr = LR,params=disc.parameters(),betas = (0.5,0.999))
optim_gen = optim.Adam(lr = LR,params=gen.parameters(),betas=(0.5,0.999))

disc.train()
gen.train()

criterion = nn.BCELoss()

fix_noise = torch.randn(BATCH_SIZE,NOISE_DIM,1,1).to(device)


writer_real = SummaryWriter()
writer_fake = SummaryWriter()

print("Starting training ...")

for epoch in range(NUM_EPOCHS):
    for i,(img,_) in enumerate(dataloader):
        img = img.to(device)
        batch = img.shape[0]

            ## train discrimnator: max log(D(x)) + log(1-D(G(z)))
            ## BCE loss minimizizes negative so its same

        disc.zero_grad()
        label = torch.full(size=(batch,),fill_value=0.9,dtype=torch.float).to(device)
    
        output = disc(img).view(-1)
        lossD_real = criterion(output,label) ##real one is going to be 
        d_x = output.mean().item()

        lossD_real.backward()
    

        noise = torch.randn(batch,NOISE_DIM,1,1).to(device)
        fake = gen(noise).to(device)
        label.fill_(0.1)

        output = disc(fake.detach()).view(-1)

        lossD_fake = criterion(output,label)
        lossD_fake.backward()


      

        lossD = lossD_fake+ lossD_real

        optim_disc.step()



        ##Train generator  max log(D(G(z)))

        gen.zero_grad() 
        
        output = disc(fake).reshape(-1).to(device)
        label.fill_(0.9) ## bce loss = -w[y.logx + (1-y).log(1-x)] if y = 1 we get -w log(x)
        loss_G = criterion(output,label)
        loss_G.backward()
        optim_gen.step()


        if i%250==0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS} , Batch {i}/{len(dataloader),} Loss Discrimnator: {lossD:.4f} , Loss Generator {loss_G:.4f}')
            
            with torch.no_grad():
                fake = gen(fix_noise).to(device)
    
                img_grid_real = torchvision.utils.make_grid(img,normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)

                writer_real.add_images(tag="real",img_tensor=img_grid_real,dataformats="CHW")
                writer_fake.add_images(tag="fake",img_tensor=img_grid_fake,dataformats="CHW")
