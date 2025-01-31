import cv2
import numpy as np
import urllib.request
import sys

def load_image(image_path):
    """Load image from file or URL."""
    if image_path.startswith("http"):
        resp = urllib.request.urlopen(image_path)
        image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def preprocess_image(image, size=(500, 500)):
    """Resize and preprocess the image to normalize features."""
    image = cv2.resize(image, size)  # Resize to standard size
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduce noise
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Enhance text
    return image

def compare_images(image1, image2):
    """Compare two images using SIFT feature matching."""
    sift = cv2.SIFT_create(nfeatures=500)  # Increase feature detection
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found. Comparison failed.")
        return 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    score = len(matches) / max(len(keypoints1), len(keypoints2))
    return score

def template_match(image, template):
    """Match a template against an image using normalized correlation."""
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val  # Higher value means better match


if __name__ == "__main__":
    img1 = load_image(sys.argv[1])
    img2 = load_image(sys.argv[2])

    if img1 is None or img2 is None:
        print("Error loading images.")
        sys.exit(1)

    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    match_score = template_match(img1, img2)
    print(f"Template Match Score: {match_score:.2f}")

    if match_score > 0.5:
        print("Fonts match closely.")
    else:
        print("Fonts do not match.")