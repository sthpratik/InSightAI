# Install dependencies in your project directory
# pip install moondream

import moondream as md
from PIL import Image

# Initialize with API key
# model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIwM2JjZGQzZS0wMmE2LTQwZTYtOTNiYy05YTI2YWY4ODM2M2YiLCJpYXQiOjE3Mzk0ODA1MTJ9.mnIFU9DhyCB-b98cLQvCwKFCyh_ORPruI1N06m0Djtc")
model = md.vl(model='./moondream-2b-int8.mf.gz')  # Initialize model

# Load an image
image = Image.open("../video/test.jpg")
encoded_image = model.encode_image(image)  # Encode image (recommended for multiple operations)


# Generate a caption (length options: "short" or "normal" (default))
caption = model.caption(encoded_image)["caption"]
print("Caption:", caption)

# Stream the caption
for chunk in model.caption(encoded_image, stream=True)["caption"]:
    print(chunk, end="", flush=True)

# Ask a question
answer = model.query(encoded_image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the answer
for chunk in model.query(encoded_image, "What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)

# Detect objects
detect_result = model.detect(image, 'subject')  # change 'subject' to what you want to detect
print("Detected objects:", detect_result["objects"])

# Point at an object
point_result = model.point(image, 'subject')  # change 'subject' to what you want to point at
print("Points:", point_result["points"])