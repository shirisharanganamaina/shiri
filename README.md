# shiri
![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/shirisharanganamaina/shiri/assets/169051602/0df0ad41-b551-4332-8cb9-818c02f77118)
![7622202030987_bounding_box.csv](https://github.com/shirisharanganamaina/shiri/files/15244437/7622202030987_bounding_box.csv)
```
import os
import csv
from PIL import Image, ImageDraw


csv_file = "/home/shirisha-ranganamaina/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987"
output_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987_with_boxes"
os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

```
