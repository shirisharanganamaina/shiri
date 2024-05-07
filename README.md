##boundingbox
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
```
##histogram
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('/home/shirisha-ranganamaina/Downloads/scripts/env.jpeg')
cv.imwrite("/home/shirisha-ranganamaina/Downloads/scripts/ro.jpg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
```
##iterate
```
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'sum' + str(sum))
    previousNum=i
```
##web
```
import cv2 
  
vid = cv2.VideoCapture(0) 
  
while(True): 
      

    ret, frame = vid.read() 
  
    
    cv2.imshow('frame', frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 

cv2.destroyAllWindows()
```
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
