import torch
from torch.utils.data import DataLoader
from dataset import TomatoDataset
from model_detector import get_detector
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES=8
EPOCHS=30
BATCH_SIZE=4

train_dataset=TomatoDataset(
    "train/images",
    "train/f-cnn_labels"
)

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x:tuple(zip(*x))
)

model=get_detector(NUM_CLASSES)
model.to(device)

optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

for epoch in range(EPOCHS):

    model.train()
    epoch_loss=0

    loop=tqdm(train_loader)

    for images,targets in loop:

        images=[img.to(device) for img in images]

        targets=[
            {k:v.to(device) for k,v in t.items()}
            for t in targets
        ]

        loss_dict=model(images,targets)

        losses=sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss+=losses.item()

        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=losses.item())

    print("Epoch Loss:",epoch_loss/len(train_loader))

torch.save(model.state_dict(),"detector_model.pth")