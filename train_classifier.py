import torch
import timm
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.cuda.empty_cache()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

dataset=ImageFolder("lesion_crops",transform)

loader=DataLoader(dataset,batch_size=8,shuffle=True)

NUM_CLASSES=7

model=timm.create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model=model.to(device)

for param in model.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

criterion=nn.CrossEntropyLoss()

EPOCHS=25

for epoch in range(EPOCHS):

    total = 0
    correct = 0
    epoch_loss = 0

    loop = tqdm(loader)

    for x,y in loop:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _,p = pred.max(1)

        total += y.size(0)
        correct += (p==y).sum().item()

        acc = correct/total

        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item(), accuracy=acc)

    print(f"\nEpoch {epoch+1} Final Accuracy: {correct/total:.4f}")

torch.save(model.state_dict(),"classifier_model.pth")