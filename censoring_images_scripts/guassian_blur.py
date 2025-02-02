import json
import cv2
import numpy as np
import time
import os


with open("/media/ydg/Windows-SSD/YDG_linux/datasets/dataset-20241108T030640Z-001/dataset/val/via_region_data.json", "r") as f:
    data = json.load(f)

image_dir = "/media/ydg/Windows-SSD/YDG_linux/datasets/dataset-20241108T030640Z-001/dataset/val"  
output_dir = "/media/ydg/Windows-SSD/YDG_linux/datasets/dataset-20241108T030640Z-001/dataset/censored_images_05"  
os.makedirs(output_dir, exist_ok=True)

times = [] 

for image_name, details in data.items():
    image_name = image_name.split(".jpg")[0] + ".jpg"

    image_path = os.path.join(image_dir, image_name)

    img = cv2.imread(image_path)
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
    else:
        print(f"File exists: {image_path}")

    if img is None:
        print(f"Could not load {image_name}")
        continue

    start_time = time.perf_counter()


    for region in details["regions"].values():
        shape = region["shape_attributes"]

        if shape["name"] != "nonviolence":  
            points = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            blurred = cv2.GaussianBlur(img, (51, 51), 0)

            img = np.where(mask[:, :, None] == 255, blurred, img)

    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img)

    end_time = time.perf_counter()
    times.append(end_time - start_time)

average_time = sum(times) / len(times)
print(f"Average processing time per image: {average_time:.4f} seconds")

