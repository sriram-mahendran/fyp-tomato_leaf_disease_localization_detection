import torch
from torch.utils.data import DataLoader
from dataset import TomatoDataset
from model_detector import get_detector
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 8

# dataset
val_dataset = TomatoDataset(
    "test/images",
    "test/f-cnn_labels"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# load model
model = get_detector(NUM_CLASSES)
model.load_state_dict(torch.load("detector_model.pth"))

model.to(device)
model.eval()

metric = MeanAveragePrecision()

with torch.no_grad():

    for images, targets in tqdm(val_loader):

        images = [img.to(device) for img in images]

        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):

            preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu()
            })

            gts.append({
                "boxes": target["boxes"],
                "labels": target["labels"]
            })

        metric.update(preds, gts)

results = metric.compute()

print("\nDetector Evaluation Results\n")

for k, v in results.items():
    print(k, ":", v)