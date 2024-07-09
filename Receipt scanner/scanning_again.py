#import necessary modules
import cv2
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import re 

from skimage.filters import threshold_local
from pytesseract import Output 
from prettytable import PrettyTable

######
##---------------HELPER FUNCTIONS---------------------##
def resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#Display grey scale image
def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

#Display RGB colour image
def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#We will use approxPolyDP for approximating more primitive contour shape consisting of as few points as possible
#Approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

#find four points of receipt
def get_receipt_contour(contours):
    for c in contours:
        approx = approximate_contour(c)
        if len(approx) == 4:
            return approx 
        
#Convert 4 points into lines / rect      
def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio
        
#Original receipt with wrapped perspective
def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (max_width, max_height))

#Threshold image
def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

##----------------------------------------------------##

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

#Load the image
receipt_path = "/Users/namishkaistha/Desktop/chipotle_receipt_desk.jpg"

#use cv2 to read the image
receipt = cv2.imread(receipt_path)

#downscale image, to make it easier to read...
## - findings - THIS WAS AN INTEGRAL STEP!! 
resize_ratio = 500 / receipt.shape[0]
og = receipt.copy()
receipt = resize(receipt, resize_ratio)
og = resize(og, resize_ratio)

#error handling
if receipt is None:
    print("File could not be read, what an L")

#grayscale the image, then apply gaussian blur 

#grayscale: to simplify the processing
grayscale_receipt = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)

# #apply gaussian blur - uses a convolution kernel/weighed kernel 
# #need to do more reading to actually understand this. - openCV documentation says to use a 5x5 filter, so that's what we use 
gaussian = cv2.GaussianBlur(grayscale_receipt, (5,5),0)

#now, run canny edge detector 
edges = cv2.Canny(gaussian, 100,200)


##detect contours - first one is source image, second is contour retrieval mode, third is contour approximation method.
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

##overlay contours 

contour_receipt_photo = cv2.drawContours(og, contours, -1, (0,255,0), 3)


##two heuristics for finding receipt contour - 1. the receipt contour is the largest contour, and 2. the contour is a rectangle 
# Get largest contour - overlay over another new copy of the receipt photo
largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
image_with_largest_contours = cv2.drawContours(receipt.copy(), largest_contours, -1, (0,255,0), 3)

#note down the four contour points 
contour_points = get_receipt_contour(largest_contours)

#make the points into a rectangle
rect = contour_to_rect(contour_points)

#grab the ORIGINAL receipt:
original = cv2.imread(receipt_path)

#restore receipt perspective - warp it so it's all you see on the screen - first, convert contour into a coordinate array - then, use rectangle points to calculate destination points of the scanned view 
# use cv2.getPerspectiveTransform to calculate transformation matrix, and finally use cv2.warpPerspective to restore the perspective
wrapped_receipt = wrap_perspective(original, rect)

bw = bw_scanner(wrapped_receipt)


#use bounding box to look for dictionary words on the wrapped black and white text 
d = pytesseract.image_to_data(bw, output_type=Output.DICT)

n_boxes = len(d['level'])
boxes = cv2.cvtColor(bw.copy(), cv2.COLOR_BGR2RGB)
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
    boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
custom_config = r'--oem 3 --psm 6'
extracted_text = pytesseract.image_to_string(wrapped_receipt, config=custom_config)
# print(extracted_text)

#things to omit
remove = ["take","out","total", "cp","card","subtotal"]

#extract letters and numbers regex 
regex = []
for line in extracted_text.splitlines():
    if re.search(r"[0-9]*\.[0-9]|[0-9]*\,[0-9]", line):
        regex.append(line)

# print(regex)

#apply the "ommitted" list
food= []
for r in regex:
    found = False
    for rem in remove:
        if rem in r.lower():
            found = True
    if found == False:
        food.append(r)


# print(food)

#A regex for food item cost 
#Food item cost regex
food_item_cost = []
for line in food:
    line = line.replace(",", ".")
    cost = re.findall(r'\d*\.?\d+|\d*,?\d+', line)  
    
    for possibleCost in cost:
        if "." in possibleCost:
            food_item_cost.append(possibleCost)
# print(food_item_cost)

#Remove cost price from food item
count = 0
only_food_items = []
for item in food:
    only_alpha = ""
    for char in item:
        if char == ']' or char == '|':
            char = 'l'
        if char.isalpha() or char.isspace():
            only_alpha += char
            
    only_alpha = re.sub(r'(?:^| )\w(?:$| )', ' ', only_alpha).strip()
    only_food_items.append(only_alpha.upper())


#Taulate Food Item and Cost
t = PrettyTable(['Item', 'Cost'])
for counter in range (0,len(food)):
    t.add_row([only_food_items[counter], food_item_cost[counter]])
print(t)





# cv2.imshow('image',boxes)

# # Wait for a key press and close the image window
# cv2.waitKey(0)
# cv2.destroyAllWindows()





