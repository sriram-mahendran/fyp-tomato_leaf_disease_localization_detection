# import os
# import cv2
# import torch
# import timm
# from torchvision import transforms
# from PIL import Image
# from model_detector import get_detector

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUM_CLASSES = 8

# # disease names
# class_names = [
#     "Early Blight",
#     "Black Spot",
#     "Healthy",
#     "Leaf Mold",
#     "Bacterial Spot",
#     "Target Spot",
#     "Late Blight"
# ]

# # ---------------- LOAD MODELS ----------------

# detector = get_detector(NUM_CLASSES)
# detector.load_state_dict(torch.load("detector_model.pth"))
# detector.to(device)
# detector.eval()

# classifier = timm.create_model("efficientnet_b4", num_classes=7)
# classifier.load_state_dict(torch.load("classifier_model.pth"))
# classifier.to(device)
# classifier.eval()

# # ---------------- TRANSFORM ----------------

# transform = transforms.Compose([
#     transforms.Resize((380,380)),
#     transforms.ToTensor()
# ])

# # ---------------- FOLDERS ----------------

# input_folder = "test/images"
# output_folder = "results"

# os.makedirs(output_folder, exist_ok=True)

# image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

# print("Total images:", len(image_files))

# # ---------------- INFERENCE LOOP ----------------

# for img_name in image_files:

#     img_path = os.path.join(input_folder, img_name)

#     img = cv2.imread(img_path)

#     if img is None:
#         print("Skipping:", img_name)
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     tensor = torch.tensor(img_rgb).permute(2,0,1).float()/255
#     tensor = tensor.unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = detector(tensor)[0]

#     boxes = outputs["boxes"]
#     scores = outputs["scores"]

#     h, w, _ = img.shape

#     for box, score in zip(boxes, scores):

#         if score < 0.2:
#             continue

#         x1, y1, x2, y2 = map(int, box.tolist())

#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)

#         crop = img_rgb[y1:y2, x1:x2]

#         if crop.size == 0:
#             continue

#         crop = Image.fromarray(crop)

#         inp = transform(crop).unsqueeze(0).to(device)

#         with torch.no_grad():
#             pred = classifier(inp)
#             cls = pred.argmax(1).item()

#         label = class_names[cls]

#         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

#         cv2.putText(
#             img,
#             f"{label} {score:.2f}",
#             (x1,y1-10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0,255,0),
#             2
#         )

#     # save result
#     save_path = os.path.join(output_folder, img_name)
#     cv2.imwrite(save_path, img)

#     print("Processed:", img_name)

# print("All images processed. Results saved in:", output_folder)



import os
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
from model_detector import get_detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 8

# disease names
class_names = [
    "Early Blight",
    "Black Spot",
    "Healthy",
    "Leaf Mold",
    "Bacterial Spot",
    "Target Spot",
    "Late Blight"
]

HEALTHY_CLASS_ID = 2

# ---------------- LOAD MODELS ----------------

detector = get_detector(NUM_CLASSES)
detector.load_state_dict(torch.load("detector_model.pth"))
detector.to(device)
detector.eval()

classifier = timm.create_model("efficientnet_b4", num_classes=7)
classifier.load_state_dict(torch.load("classifier_model.pth"))
classifier.to(device)
classifier.eval()

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((380,380)),
    transforms.ToTensor()
])

# ---------------- FOLDERS ----------------

input_folder = "test/images"
output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

print("Total images:", len(image_files))

# ---------------- INFERENCE LOOP ----------------

for img_name in image_files:

    img_path = os.path.join(input_folder, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print("Skipping:", img_name)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.tensor(img_rgb).permute(2,0,1).float()/255
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = detector(tensor)[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]

    h, w, _ = img.shape

    detections = []

    # ---------------- CLASSIFICATION ----------------

    for box, score in zip(boxes, scores):

        if score < 0.2:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = Image.fromarray(crop)

        inp = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = classifier(inp)
            cls = pred.argmax(1).item()

        detections.append({
            "box": (x1,y1,x2,y2),
            "score": score.item(),
            "class": cls
        })

    # ---------------- REMOVE HEALTHY IF DISEASE EXISTS ----------------

    has_disease = any(d["class"] != HEALTHY_CLASS_ID for d in detections)

    if has_disease:
        detections = [d for d in detections if d["class"] != HEALTHY_CLASS_ID]

    # ---------------- DRAW BOXES ----------------

    for d in detections:

        x1,y1,x2,y2 = d["box"]
        score = d["score"]
        cls = d["class"]

        label = class_names[cls]

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            img,
            f"{label} {score:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # save result
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)

    print("Processed:", img_name)

print("All images processed. Results saved in:", output_folder)