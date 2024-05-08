# shiri
![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/shirisharanganamaina/shiri/assets/169051602/0df0ad41-b551-4332-8cb9-818c02f77118)
![7622202030987_bounding_box.csv](https://github.com/shirisharanganamaina/shiri/files/15244437/7622202030987_bounding_box.csv)
```
import os
import csv
from PIL import Image, ImageDraw
    import os: This imports the 'os' module, which provides a way of using operating system-dependent functionality. In this script, it's used to create directories and handle file paths.
    import csv: This imports the 'csv' module, which provides functionality to read and write CSV files.
    from PIL import Image, ImageDraw: This imports the 'Image' and 'ImageDraw' modules from the Python Imaging Library (PIL). These modules are used for working with images and drawing on them.

csv_file = "/home/shirisha-ranganamaina/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987"
output_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987_with_boxes"
os.makedirs(output_dir, exist_ok=True)

csv_file, image_dir, output_dir: These variables store the paths to the CSV file containing bounding box coordinates, the directory containing the images, and the directory where the output images with bounding boxes will be saved respectively.

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
os.makedirs(output_dir, exist_ok=True): This line creates the output directory specified by output_dir if it doesn't exist already. The exist_ok=True argument ensures that the operation doesn't raise an error if the directory already exists.
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
Inside the loop over csv_reader, each row is processed:

    image_name is extracted from the 'filename' column of the CSV.
    The full path of the input image (image_path) and the output image (output_path) are constructed using os.path.join.
    The input image is opened using Image.open(image_path).
```
##histogram
![env](https://github.com/shirisharanganamaina/shiri/assets/169051602/f5150ee2-7b3a-4d9a-a169-982e52b49c6e)
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np: Imports the numpy library, commonly used for numerical operations, and assigns it the alias np.
    import cv2 as cv: Imports the OpenCV library and assigns it the alias cv.
    from matplotlib import pyplot as plt: Imports the pyplot module from the matplotlib library and assigns it the alias plt. This module is used to create plots.
 # Load an image
img = cv.imread('/home/shirisha-ranganamaina/Downloads/scripts/env.jpeg')
cv.imwrite("/home/shirisha-ranganamaina/Downloads/scripts/ro.jpg",img)
    cv.imread('/home/shirisha-ranganamaina/Downloads/scripts/env.jpeg'): Reads the image file 'env.jpeg' located at the specified path using OpenCV's imread function and stores it in the variable img.
    cv.imwrite("/home/shirisha-ranganamaina/Downloads/scripts/ro.jpg",img): Writes the image img to the file 'ro.jpg' at the specified path using OpenCV's imwrite function. This line seems unnecessary and could be removed.
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
# Calculate histogram
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()    assert img is not None, "file could not be read, check with os.path.exists()": This line checks if the image was successfully read. If img is None, it raises an AssertionError with the message "file could not be read, check with os.path.exists()".
    color = ('b','g','r'): Defines a tuple of color channel identifiers: blue, green, and red.
    The code then iterates over each color channel:
        cv.calcHist([img],[i],None,[256],[0,256]): Calculates the histogram for the i-th color channel (i=0 for blue, i=1 for green, i=2 for red) using OpenCV's calcHist function. It computes a histogram with 256 bins in the range [0,256).
        plt.plot(histr,color = col): Plots the histogram histr with the specified color col.
        plt.xlim([0,256]): Sets the x-axis limits of the plot to [0,256) to match the range of pixel intensities.
    Finally, plt.show() displays the plot containing all three histograms.
```
