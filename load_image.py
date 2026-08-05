import cv2

# Path to your sample image
image_path = "data/raw_images/sample.png"

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Check the file path.")
else:
    print("Image loaded successfully!")
    print(f"Image shape (height, width, channels): {image.shape}")