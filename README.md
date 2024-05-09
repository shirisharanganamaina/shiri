##boundingbox

```
import os
import csv
from PIL import Image, ImageDraw

Import Statements:
import os: This imports the 'os' module, which provides a way of using operating system-dependent functionality. In this script, it's used to create directories and handle file paths.
import csv: This imports the 'csv' module, which provides functionality to read and write CSV files.
from PIL import Image, ImageDraw: This imports the 'Image' and 'ImageDraw' modules from the Python Imaging Library (PIL). These modules are used for working with images and drawing on them.
```
![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/shirisharanganamaina/shiri/assets/169051602/34f47053-1c1a-4b37-8e96-68135cb3af05)

csv_file = "/home/shirisha-ranganamaina/Downloads/7622202030987_bounding_box.csv"

image_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987"

output_dir = "/home/shirisha-ranganamaina/Downloads/7622202030987_with_boxes"


csv_file, image_dir, output_dir: These variables store the paths to the CSV file containing bounding box coordinates, the directory containing the images, and the directory where the output images with bounding boxes will be saved respectively.


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
    

![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/shirisharanganamaina/shiri/assets/169051602/0df0ad41-b551-4332-8cb9-818c02f77118)

Processing CSV File:
with open(csv_file, 'r') as file:: This opens the CSV file specified by csv_file in read mode.
csv_reader = csv.DictReader(file): This creates a CSV reader object which will iterate over the rows of the CSV file, treating each row as a dictionary where the keys are the column headers.

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


Function Definitions:
 draw_boxes(image, boxes): This function takes an image and a list of dictionaries representing bounding boxes and draws these bounding boxes on the image using red outlines.
crop_image(image, boxes): This function takes an image and a list of dictionaries representing bounding boxes and crops the image to extract regions defined by these bounding boxes. It returns a list of cropped images.


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

output
```
![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/shirisharanganamaina/shiri/assets/169051602/cff798f2-8a70-4e13-9f58-266696fbbafa)
```

##histogram

```
import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

    import numpy as np: Imports the numpy library, commonly used for numerical operations, and assigns it the alias np.
    import cv2 as cv: Imports the OpenCV library and assigns it the alias cv.
    from matplotlib import pyplot as plt: Imports the pyplot module from the matplotlib library and assigns it the alias plt. This module is used to create plots.
``
 ![env](https://github.com/shirisharanganamaina/shiri/assets/169051602/f5150ee2-7b3a-4d9a-a169-982e52b49c6e)

img = cv.imread('/home/shirisha-ranganamaina/Downloads/scripts/env.jpeg')
cv.imwrite("/home/shirisha-ranganamaina/Downloads/scripts/ro.jpg",img)

cv.imread('/home/shirisha-ranganamaina/Downloads/scripts/env.jpeg'): Reads the image file 'env.jpeg' located at the specified path using OpenCV's imread function and stores it in the variable img.


assert img is not None, "file could not be read, check with os.path.exists()"

color = ('b','g','r')

for i,col in enumerate(color):

 histr = cv.calcHist([img],[i],None,[256],[0,256])
 
 plt.plot(histr,color = col)
 
 plt.xlim([0,256])
 
plt.show()


 assert img is not None, "file could not be read, check with os.path.exists()": This line checks if the image was successfully read. If img is None, it raises an AssertionError with the message "file could not be read, check with os.path.exists()".
color = ('b','g','r'): Defines a tuple of color channel identifiers: blue, green, and red.
The code then iterates over each color channel:
cv.calcHist([img],[i],None,[256],[0,256]): Calculates the histogram for the i-th color channel (i=0 for blue, i=1 for green, i=2 for red) using OpenCV's calcHist function. It computes a histogram with 256 bins in the range [0,256).
 plt.plot(histr,color = col): Plots the histogram histr with the specified color col.
plt.xlim([0,256]): Sets the x-axis limits of the plot to [0,256) to match the range of pixel intensities.
Finally, plt.show() displays the plot containing all three histograms.

```
output
```
![Screenshot from 2024-05-06 19-01-44](https://github.com/shirisharanganamaina/shiri/assets/169051602/e36e8db2-b618-464b-97d6-c8eecc079561)
```


##Iterate

```
num = list(range(10))

previousNum = 0

for i in num:


ist(range(10)): Creates a list containing integers from 0 to 9 (excluding 10) using the range() function and converts it to a list. So, num becomes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].


 sum = previousNum + i
 print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'sum' + str(sum))
 previousNum=i

sum = previousNum + i: Calculates the sum of the current number (i) and the previous number (previousNum).
print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'sum' + str(sum)): Prints the current number (i), the previous number (previousNum),       and their sum (sum) in a formatted string.
previousNum = i: Updates the value of previousNum to the current number i for the next iteration.
```

output

Current Number 0Previous Number 0sum0

Current Number 1Previous Number 0sum1

Current Number 2Previous Number 1sum3

Current Number 3Previous Number 2sum5

Current Number 4Previous Number 3sum7

Current Number 5Previous Number 4sum9

Current Number 6Previous Number 5sum11

Current Number 7Previous Number 6sum13

Current Number 8Previous Number 7sum15

Current Number 9Previous Number 8sum17
```
##Web

This imports the OpenCV library. OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library.
``` import cv2 
  vid = cv2.VideoCapture(0)
  while(True): 
  ret, frame = vid.read() 
   cv2.imshow('frame', frame) 
     if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
cv2.destroyAllWindows()

opens the default camera (usually the webcam) connected to the computer. You can specify a different camera index if multiple cameras are available.
This starts an infinite loop to continuously capture frames from the video feed.
vid.read() reads a frame from the video capture object vid. It returns a Boolean value ret (True if the frame is read correctly) and the captured frame frame.
This checks for the 'q' key press event. If the user presses the 'q' key, the loop breaks, and the program exits.
 cv2.waitKey(1) waits for a key event for 1 millisecond. It returns the ASCII value of the key pressed.
0xFF == ord('q') checks if the key pressed is 'q'.This closes all OpenCV windows created during the program execution. It's a good practice to clean up resources and close windows before exiting the program.
web vedio
```
!https://github.com/shirisharanganamaina/shiri/assets/169051602/b6fe8cf8-ef60-4d7c-b422-60ddc3d19be4
```
