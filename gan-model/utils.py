import torch
x = torch.full(size=(64,),fill_value=1)
x.fill_(0)
print(x)