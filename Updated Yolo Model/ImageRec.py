import os
import time
import shutil
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO  # Import the YOLOv8 model from ultralytics
import numpy as np
from PIL import Image
import io
import sys
import cv2

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model with the new weights
model = YOLO('/Users/alanlee/documents/sc2079/Tester/final_best.pt')

# Define the new class names from your updated model
new_model_class_names = {
    0: '1',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: 'A',
    10: 'B',
    11: 'Bullseye',
    12: 'C',
    13: 'D',
    14: 'Down',
    15: 'E',
    16: 'F',
    17: 'G',
    18: 'H',
    19: 'Left',
    20: 'Right',
    21: 'S',
    22: 'Stop',
    23: 'T',
    24: 'U',
    25: 'Up',
    26: 'V',
    27: 'W',
    28: 'X',
    29: 'Y',
    30: 'Z',
    # Class ID 31 ('z-') can be ignored
    31: 'z-'
}

# Update the custom class name mapping to match the new class names
custom_class_name_mapping = {
    # Numbers
    '1': 'Number One',
    '2': 'Number Two',
    '3': 'Number Three',
    '4': 'Number Four',
    '5': 'Number Five',
    '6': 'Number Six',
    '7': 'Number Seven',
    '8': 'Number Eight',
    '9': 'Number Nine',

    # Alphabets
    'A': 'Alphabet A',
    'B': 'Alphabet B',
    'C': 'Alphabet C',
    'D': 'Alphabet D',
    'E': 'Alphabet E',
    'F': 'Alphabet F',
    'G': 'Alphabet G',
    'H': 'Alphabet H',
    'S': 'Alphabet S',
    'T': 'Alphabet T',
    'U': 'Alphabet U',
    'V': 'Alphabet V',
    'W': 'Alphabet W',
    'X': 'Alphabet X',
    'Y': 'Alphabet Y',
    'Z': 'Alphabet Z',

    # Arrows
    'Up': 'Up arrow',
    'Down': 'Down arrow',
    'Right': 'Right arrow',
    'Left': 'Left arrow',

    # Stop Sign
    'Stop': 'Stop'
    # Note: 'Bullseye' is not mapped, as per your instruction
}

# Update the class-to-image ID mapping
class_to_image_id = {
    # Numbers
    'Number One': 11,
    'Number Two': 12,
    'Number Three': 13,
    'Number Four': 14,
    'Number Five': 15,
    'Number Six': 16,
    'Number Seven': 17,
    'Number Eight': 18,
    'Number Nine': 19,

    # Alphabets
    'Alphabet A': 20,
    'Alphabet B': 21,
    'Alphabet C': 22,
    'Alphabet D': 23,
    'Alphabet E': 24,
    'Alphabet F': 25,
    'Alphabet G': 26,
    'Alphabet H': 27,
    'Alphabet S': 28,
    'Alphabet T': 29,
    'Alphabet U': 30,
    'Alphabet V': 31,
    'Alphabet W': 32,
    'Alphabet X': 33,
    'Alphabet Y': 34,
    'Alphabet Z': 35,

    # Arrows
    'Up arrow': 36,
    'Down arrow': 37,
    'Right arrow': 38,
    'Left arrow': 39,

    # Stop Sign
    'Stop': 40
}

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def predict_image_week_9(image_path, model, obstacle_id=None):
    """
    Processes the image using the YOLOv8 model, draws a bounding box around the closest detected object,
    and returns a JSON response with the provided obstacle_id (if any) and detected image_id.
    The label on the image will always show the predicted obstacle ID (detected from the image).
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if img is None:
        return {"error": "Failed to load image"}

    # Use the YOLOv8 model to predict objects in the image
    results = model(img, conf=0.25)  # Set confidence threshold to 0.25 for more detections

    detected_class_name = None    # Initialize to store the detected class name
    image_id = None               # Initialize the image ID
    largest_area = 0              # Initialize variable to track the largest area
    closest_box = None            # Initialize variable to store the closest bounding box
    check_detection_found = False # Flag to track if any valid detection was found

    # Check if any detections were found
    if results is None or len(results) == 0 or len(results[0].boxes) == 0:
        print("No detections were made.")
        return {"error": "No detections found"}

    # Iterate over each result (bounding box, class, confidence, etc.)
    for result in results:
        for box in result.boxes:

            class_idx = int(box.cls[0].cpu().numpy())  # Class index

            # Ignore class ID 31 ('z-')
            if class_idx == 31:
                continue

            # Get the model class name from the class index
            model_class_name = new_model_class_names.get(class_idx)

            # Skip if class name is not found
            if model_class_name is None:
                continue

            # Map to custom class name
            original_class_name = custom_class_name_mapping.get(model_class_name, model_class_name)

            # Skip "Bullseye" class
            if original_class_name == "Bullseye":
                continue

            check_detection_found = True  # A valid detection is found

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            area = (x2 - x1) * (y2 - y1)  # Calculate the area of the bounding box

            if area > largest_area:
                largest_area = area
                closest_box = box

                detected_class_name = original_class_name
                image_id = class_to_image_id.get(detected_class_name, None)

    # If no valid detection was found (only "Bullseye" or no detections), return "No detections found"
    if not check_detection_found:
        print("Only Bullseye detected or no valid detections.")
        return {"error": "No detections found"}

    # The obstacle_id to return is the one provided (if any), else the detected class name
    obstacle_id_to_return = obstacle_id if obstacle_id else detected_class_name

    # Draw the bounding box and save the image
    if closest_box:
        x1, y1, x2, y2 = closest_box.xyxy[0].cpu().numpy()

        # Draw the bounding box in green
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)  # Thicker box

        # Create the label text with two lines: detected class name and image ID
        label_line_1 = f"{detected_class_name}"
        label_line_2 = f"Image id={image_id}"

        # Set up font properties for OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Calculate the size of the text to create a white background box behind the text
        (label_width_1, label_height_1), _ = cv2.getTextSize(label_line_1, font, font_scale, font_thickness)
        (label_width_2, label_height_2), _ = cv2.getTextSize(label_line_2, font, font_scale, font_thickness)
        label_width = max(label_width_1, label_width_2)  # Get the maximum width of the two lines

        # Calculate where to position the label box and text
        label_height = label_height_1 + label_height_2 + 10  # Total label height with some padding

        # Draw a white rectangle behind the text for better readability
        cv2.rectangle(img, (int(x1), int(y1) - label_height - 10), (int(x1) + label_width + 10, int(y1)), (255, 255, 255), cv2.FILLED)

        # Draw the detected class name and image ID inside the rectangle
        cv2.putText(img, label_line_1, (int(x1) + 5, int(y1) - label_height + label_height_1), font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(img, label_line_2, (int(x1) + 5, int(y1) - 5), font, font_scale, (0, 0, 0), font_thickness)

        # Save the image with bounding box and labels
        output_path = os.path.join('predictions', f"image_{image_id}.jpg" if image_id else "unknown_image.jpg")
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        cv2.imwrite(output_path, img)

    # Return the result with obstacle_id (provided or detected) and image_id as a string
    return {
        "obstacle_id": obstacle_id_to_return,
        "image_id": str(image_id) if image_id else None
    }


def stitch_image():
    """
    Stitches the images in the 'predictions' folder together and saves the stitched image.
    Adjusted to fit up to 8 images by arranging them in rows of 4 images per row.
    """
    imgFolder = 'predictions'  # Folder to look for stitched images
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Get all the images in the predictions folder
    imgPaths = glob.glob(os.path.join(imgFolder, "*.jpg"))
    print(f"Found images for stitching in 'predictions': {imgPaths}")  # Debugging line

    if not imgPaths:
        print("No images found to stitch in 'predictions'.")
        return None

    # Open the images
    images = [Image.open(x) for x in imgPaths]

    # Get the width and height of individual images (assuming all images have the same size)
    image_width, image_height = images[0].size

    # Set number of images per row
    images_per_row = 4  # Adjust this number to change how many images per row

    # Calculate the total width and height for the stitched image
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
    total_width = images_per_row * image_width  # Total width based on images per row
    total_height = num_rows * image_height     # Total height based on number of rows

    # Create a new blank image with the calculated total width and height
    stitched_img = Image.new('RGB', (total_width, total_height))

    # Paste images into the stitched image in the correct position
    x_offset = 0
    y_offset = 0

    for i, img in enumerate(images):
        stitched_img.paste(img, (x_offset, y_offset))

        # Move to the next column
        x_offset += image_width

        # If we've placed 'images_per_row' images in a row, move to the next row
        if (i + 1) % images_per_row == 0:
            x_offset = 0  # Reset x_offset to start a new row
            y_offset += image_height  # Move y_offset down to the next row

    # Save the stitched image
    stitched_img.save(stitchedPath)
    print(f"Stitched image saved at: {stitchedPath}")  # Debugging line

    return stitched_img



# Function to stitch images from the 'own_results' folder
def stitch_image_own():
    """
    Stitches the images in the 'own_results' folder together and saves the stitched image.
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder, "annotated_image_*.jpg"))
    if not imgPaths:
        print("No images found to stitch.")
        return None

    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)
    stitchedImg.save(stitchedPath)

    print(f"Stitched image saved at: {stitchedPath}")  # Debugging line
    return stitchedImg

# Flask route for image prediction
@app.route('/image', methods=['POST'])
def image_predict():
    try:
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "Missing file in request"}), 400

        file = request.files['file']
        filename = file.filename

        # Save the image file to a specific location
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Get obstacle_id from request arguments if provided
        obstacle_id = request.form.get('obstacle_id', None)

        # Call the predict_image_week_9 function with the file path, model, and obstacle_id
        result = predict_image_week_9(file_path, model, obstacle_id=obstacle_id)

        # Optional: Remove the uploaded file if not needed
        os.remove(file_path)

        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# Route to trigger stitching from the 'predictions' folder
@app.route('/stitch', methods=['GET'])
def stitch():
    img = stitch_image()
    img2 = stitch_image_own()
    return jsonify({"result": "Stitching completed"})

# Flask route to check the status
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"result": "ok"})

# Ensure necessary directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
    print("Created 'uploads' directory.")

if not os.path.exists('predictions'):
    os.makedirs('predictions')
    print("Created 'predictions' directory.")

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=5001, debug=True)
