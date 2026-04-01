import os
import cv2

img_dir="train/images"
label_dir="train/f-cnn_labels"

crop_dir="lesion_crops"

os.makedirs(crop_dir,exist_ok=True)

for file in os.listdir(img_dir):

    img=cv2.imread(os.path.join(img_dir,file))

    label_file=file.replace(".jpg",".txt").replace(".png",".txt")

    label_path=os.path.join(label_dir,label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:

        for i,line in enumerate(f):

            c,x1,y1,x2,y2=map(int,line.split())

            crop=img[y1:y2,x1:x2]

            class_dir=os.path.join(crop_dir,str(c))
            os.makedirs(class_dir,exist_ok=True)

            cv2.imwrite(
                os.path.join(class_dir,f"{file}_{i}.jpg"),
                crop
            )