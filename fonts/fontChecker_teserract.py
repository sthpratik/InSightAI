import cv2
import numpy as np
import urllib.request
import sys

import pytesseract

def extract_text(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image, config="--psm 6")


def load_image(image_path):
    """Load image from file or URL."""
    if image_path.startswith("http"):
        resp = urllib.request.urlopen(image_path)
        image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def compare_images(image1, image2):

    text1 = extract_text(img1)
    text2 = extract_text(img2)
    print(text1)
    print(text2)
    if text1 == text2:
        print("The text in both images matches, fonts might be similar.")
    else:
        print("Texts differ, fonts may not be the same.")
   

if __name__ == "__main__":
    img1 = load_image(sys.argv[1])
    img2 = load_image(sys.argv[2])

    if img1 is None or img2 is None:
        print("Error loading images.")
        sys.exit(1)

    match_score = compare_images(img1, img2)
