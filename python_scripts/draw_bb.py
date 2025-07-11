import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == "__main__":
    try:
        img = Image.open("original_scans/Reel169/ViewScan_400000.tif")
        image_np = np.array(img)
    except FileNotFoundError:
        print("Error: Image file not found. Please check the path.")
        exit()

    # Display the image
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    plt.axis('off')  # Hide axes for a cleaner view

    # Example bounding box: (x_start, y_start), width, height
    # Adjust these values based on your desired bounding box location and size
    x_min, y_min, width, height = 10, 0, 330, 600
    rect = patches.Rectangle((x_min, y_min), width, height,
                             linewidth=1, edgecolor='r', facecolor='none')
    
    ax.add_patch(rect)

    plt.show()