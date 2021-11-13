from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
help="path to input receipt image")
ap.add_argument("-d", "--debug", type=int, default=-1,
help="whether or not we are visualizing each stpe of the pipeline")
# ap.add_argument("-a", "--tessdata-dir", default="/home/moe/Desktop/Python_Project/OCR/ara")
args = vars(ap.parse_args())

# tessdata_dir_config = '--tessdata-dir "/home/moe/Desktop/Python_Project/OCR/ara.traineddata"'

# load the input image from disk, resize it, and compute the ratio
# of the *new* width to the *old* width
orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

# convert the image to grayscale, blur it slightly, and then apply
# edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 75, 200)

# check to see if we should show the output of our edge detection
# procedure
if args["debug"] > 0:
    cv2.imshow("Input", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

# find contours in the edge map and sort them by size in descending
# order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


receiptCnt = None

# loop over the contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        receiptCnt = approx
        break
if receiptCnt is None:
    raise Exception(("Could not find receipt outline. "
    "Try debugging your edge detection and contour steps."))

if args["debug"] > 0:
    output = image.copy()
    cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Receipt Outline", output)
    cv2.waitKey(0)

receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

# show transformed image
cv2.imshow("Receipt Transform", imutils.resize(receipt, width=500))
cv2.waitKey(0)

# the text is *concatenated across the row* (additionally, for your
# own images you may need to apply additional processing to cleanup
# the image, including resizing, thresholding, etc.)
options = "--psm 4"
text = pytesseract.image_to_string(
    cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB),
    config=options)

# show the raw output of the OCR process
print("[INFO] raw output:")
print("==================")
print(text)
print("\n")

# define a regular expression that will match line items that include
# a price component
pricePattern = r'([0-9]+\.[0-9]+)'

# show the output of filtering out *only* the line items in the
# receipt
print("[INFO] price line items:")
print("========================")

# loop over each of the line items in the OCR'd receipt
for row in text.split("\n"):
# check to see if the price regular expression matches the current
# row
    if re.search(pricePattern, row) is not None:
        print(row)