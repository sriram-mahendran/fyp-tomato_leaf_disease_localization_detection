import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

images_dir   = "train/images"
labels_dir   = "train/f-cnn_labels"

class_names = [
    "Early Blight",
    "Black Spot",
    "Healthy",
    "Leaf Mold",
    "Bacterial Spot",
    "Target Spot",
    "Late Blight"
]

NUM_IMAGES = 4          
valid_pairs = []

for fname in os.listdir(images_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    base = os.path.splitext(fname)[0]
    label_file = os.path.join(labels_dir, base + ".txt")
    
    if os.path.isfile(label_file):
        valid_pairs.append((os.path.join(images_dir, fname), label_file))

if not valid_pairs:
    print("No images with matching label files found.")
else:
    print(f"Found {len(valid_pairs)} images with labels")

if len(valid_pairs) >= NUM_IMAGES:
    selected = random.sample(valid_pairs, NUM_IMAGES)
else:
    selected = valid_pairs
    print(f"Showing all available {len(selected)} images")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()   # flatten to 1D array

for i, (img_path, label_path) in enumerate(selected):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    ax = axes[i]
    ax.imshow(img)
    
    # Read labels
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(class_names):
                        xmin, ymin, xmax, ymax = map(int, parts[1:5])
                        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
                        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
                        boxes.append((class_names[cls_id], xmin, ymin, xmax, ymax))
                except:
                    pass
    
    # Draw boxes + labels
    for class_name, xmin, ymin, xmax, ymax in boxes:
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2.4,
            edgecolor='yellow',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Text position: above if possible, otherwise inside
        text_y = ymin - 22 if ymin > 35 else ymin + (ymax - ymin) + 12
        ax.text(
            xmin + 4,
            text_y,
            class_name,
            color='black',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.7, pad=1.8, edgecolor='none')
        )
    
    ax.set_title(os.path.basename(img_path), fontsize=11)
    ax.axis('off')

plt.suptitle("Random Images with Bounding Boxes & Class Names", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# import os
# import random
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # ─── Configuration ────────────────────────────────────────
# dataset_path = "train"
# images_dir   = os.path.join(dataset_path, "images")
# labels_dir   = os.path.join(dataset_path, "labels")

# NUM_SAMPLES  = 12          # how many images to show
# IMG_SIZE     = (8, 6)      # figure size per image (width, height)

# # ─── Collect image paths that have matching label files ───
# image_paths = []

# for fname in os.listdir(images_dir):
#     if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#         continue
    
#     label_path = os.path.join(labels_dir, fname.rsplit('.', 1)[0] + '.txt')
#     if os.path.isfile(label_path):
#         image_paths.append(os.path.join(images_dir, fname))

# if not image_paths:
#     print("No images with matching label files found!")
# else:
#     print(f"Found {len(image_paths)} images with labels")

# # ─── Pick random samples ──────────────────────────────────
# if len(image_paths) > NUM_SAMPLES:
#     selected_paths = random.sample(image_paths, NUM_SAMPLES)
# else:
#     selected_paths = image_paths
#     print(f"Showing all {len(selected_paths)} available images")

# # ─── Visualization ────────────────────────────────────────
# plt.figure(figsize=(IMG_SIZE[0]*4, IMG_SIZE[1]*3))  # 4 columns × 3 rows example

# for i, img_path in enumerate(selected_paths, 1):
#     # load image
#     img = cv2.imread(img_path)
#     if img is None:
#         continue
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h, w = img.shape[:2]
    
#     # read label file
#     base_name = os.path.basename(img_path).rsplit('.', 1)[0]
#     label_path = os.path.join(labels_dir, base_name + '.txt')
    
#     boxes = []
#     if os.path.isfile(label_path):
#         with open(label_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue
#                 cls_id, x_center, y_center, bw, bh = map(float, parts[:5])
#                 # convert YOLO format → x_min, y_min, width, height
#                 x_min = (x_center - bw/2) * w
#                 y_min = (y_center - bh/2) * h
#                 box_w = bw * w
#                 box_h = bh * h
#                 boxes.append((cls_id, x_min, y_min, box_w, box_h))
    
#     # plot
#     ax = plt.subplot(3, 4, i)
#     ax.imshow(img)
    
#     # draw boxes + class text
#     for cls_id, x, y, bw, bh in boxes:
#         rect = patches.Rectangle(
#             (x, y), bw, bh,
#             linewidth=2.2, edgecolor='lime', facecolor='none'
#         )
#         ax.add_patch(rect)
        
#         # class name (using class ID for now — see note below)
#         label_text = f"class {int(cls_id)}"
#         ax.text(x, y-8, label_text,
#                 color='white', fontsize=9, fontweight='bold',
#                 bbox=dict(facecolor='lime', alpha=0.7, pad=1.8, edgecolor='none'))
    
#     ax.set_title(os.path.basename(img_path), fontsize=10)
#     ax.axis('off')

# plt.suptitle("Random Images with YOLO Bounding Boxes + Class IDs", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()



# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import random

# # ─── Configuration ───────────────────────────────────────────────────────
# images_dir = "train/images"
# labels_dir = "train/labels"

# # Pick one image that has a label (change this!)
# example_image_name = "IMG_0257_JPG.rf.97391f1ec271ff57de68db2e661461bb.jpg"          # ← REPLACE with real filename
# # Or random selection:
# # all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
# # example_image_name = random.choice(all_images)

# img_path = os.path.join(images_dir, example_image_name)
# label_path = os.path.join(labels_dir, os.path.splitext(example_image_name)[0] + ".txt")

# # ─── Load original image and labels ──────────────────────────────────────
# img_orig = cv2.imread(img_path)
# if img_orig is None:
#     raise FileNotFoundError(f"Cannot read image: {img_path}")

# img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
# h, w = img_orig.shape[:2]

# boxes = []  # [class_id, x_center, y_center, width, height] normalized
# if os.path.exists(label_path):
#     with open(label_path) as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 5:
#                 cls, xc, yc, bw, bh = map(float, parts[:5])
#                 boxes.append([int(cls), xc, yc, bw, bh])

# # ─── Prepare 4 versions ──────────────────────────────────────────────────
# versions = []

# # 1. Original
# versions.append({
#     'img': img_orig.copy(),
#     'boxes': [b[:] for b in boxes],           # deep copy
#     'title': "Original"
# })

# # 2. Horizontal Flip
# img_hflip = cv2.flip(img_orig, 1)
# boxes_hflip = []
# for cls, xc, yc, bw, bh in boxes:
#     boxes_hflip.append([cls, 1.0 - xc, yc, bw, bh])
# versions.append({
#     'img': img_hflip,
#     'boxes': boxes_hflip,
#     'title': "Horizontal Flip"
# })

# # 3. Rotation (±15°)
# angle = random.uniform(-15, 15)
# M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
# img_rot = cv2.warpAffine(img_orig, M, (w, h),
#                          flags=cv2.INTER_LINEAR,
#                          borderMode=cv2.BORDER_CONSTANT,
#                          borderValue=(114,114,114))

# boxes_rot = []
# cos_a = np.cos(np.radians(angle))
# sin_a = np.sin(np.radians(angle))
# for cls, xc, yc, bw, bh in boxes:
#     cx = xc * w
#     cy = yc * h
#     dx = cx - w/2
#     dy = cy - h/2
#     new_dx = dx * cos_a - dy * sin_a
#     new_dy = dx * sin_a + dy * cos_a
#     new_cx = new_dx + w/2
#     new_cy = new_dy + h/2
#     # small angle → width & height approx unchanged
#     boxes_rot.append([cls, new_cx/w, new_cy/h, bw, bh])

# versions.append({
#     'img': img_rot,
#     'boxes': boxes_rot,
#     'title': f"Rotation {angle:.1f}°"
# })

# # 4. Vertical Flip
# img_vflip = cv2.flip(img_orig, 0)
# boxes_vflip = []
# for cls, xc, yc, bw, bh in boxes:
#     boxes_vflip.append([cls, xc, 1.0 - yc, bw, bh])
# versions.append({
#     'img': img_vflip,
#     'boxes': boxes_vflip,
#     'title': "Vertical Flip"
# })

# # ─── Draw all 4 in one figure (2×2 grid) ─────────────────────────────────
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flat  # make it 1D for easy indexing

# for i, ver in enumerate(versions):
#     ax = axes[i]
#     ax.imshow(ver['img'])
    
#     for cls_id, xc, yc, bw, bh in ver['boxes']:
#         x_min = int((xc - bw/2) * w)
#         y_min = int((yc - bh/2) * h)
#         box_w = int(bw * w)
#         box_h = int(bh * h)
        
#         rect = patches.Rectangle(
#             (x_min, y_min), box_w, box_h,
#             linewidth=2.5, edgecolor='lime', facecolor='none'
#         )
#         ax.add_patch(rect)
        
#         ax.text(x_min, y_min - 12 if y_min > 20 else y_min + box_h + 5,
#                 f"cls {cls_id}",
#                 color='white', fontsize=10, fontweight='bold',
#                 bbox=dict(facecolor='lime', alpha=0.7, pad=2, edgecolor='none'))
    
#     ax.set_title(ver['title'], fontsize=13)
#     ax.axis('off')

# plt.suptitle(f"YOLO Augmentations – {example_image_name}", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()