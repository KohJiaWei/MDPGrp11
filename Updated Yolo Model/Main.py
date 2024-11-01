import os
import torch
from ultralytics import YOLO

def adjust_annotations(annotation_dir, mapping):
    """
    Adjust annotation files to reflect new class IDs based on a mapping.
    
    :param annotation_dir: Directory containing annotation files
    :param mapping: Dictionary mapping old class IDs to new class IDs
    """
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(annotation_dir, filename), 'r') as file:
                lines = file.readlines()

            with open(os.path.join(annotation_dir, filename), 'w') as file:
                for line in lines:
                    parts = line.split()
                    old_class_id = int(parts[0])
                    new_class_id = mapping.get(old_class_id, old_class_id)  # Map old class ID to new class ID
                    file.write(f"{new_class_id} {' '.join(parts[1:])}\n")

def main():
    # Check if GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pre-trained YOLO model
    model = YOLO('C:\\Users\\Alan\\Desktop\\sc2079\\model22\\model.pt').to(device)  # Load a pre-trained YOLOv8 model

    # Manually define the mapping from old class IDs to new class IDs
    old_to_new_mapping = {
        0: 0,  # 'AlphabetA' -> 'A'
        1: 1,  # 'AlphabetB' -> 'B'
        2: 2,  # 'AlphabetC' -> 'Bullseye'
        3: 3,  # 'AlphabetD' -> 'C'
        4: 4,  # 'AlphabetE' -> 'D'
        5: 5,  # 'AlphabetF' -> 'Down' (dummy)
        6: 6,  # 'AlphabetG' -> 'E'   (Dummy)
        7: 7,  # 'AlphabetH' -> 'Eight' (dummy)
        8: 8, # 'AlphabetS' -> 'S'
        9: 9, # 'AlphabetT' -> 'T'
        10: 10, # 'AlphabetU' -> 'U'
        11: 11, # 'AlphabetV' -> 'V'
        12: 12, # 'AlphabetW' -> 'W'
        13: 13, # 'AlphabetX' -> 'X'
        14: 14, # 'AlphabetY' -> 'Y'
        15: 15, # 'AlphabetZ' -> 'Z'
        16: 16, # 'Stop' -> 'Stop'
        17: 27,  # 'bullseye' -> 'Bullseye'
        18: 17,  # 'down arrow' -> 'Down'
        19: 18,  # 'eight' -> 'Eight'
        20: 19,  # 'five' -> 'Five'
        21: 20, # 'four' -> 'Four'
        22: 21, # 'left arrow' -> 'Left'
        23: 22, # 'nine' -> 'Nine'
        24: 23, # 'one' -> 'One'
        25: 24, # 'right arrow' -> 'Right'
        26: 25, # 'seven' -> 'Seven'
        27: 26, # 'six' -> 'Six'
        28: 28, # 'three' -> 'Three'
        29: 29, # 'two' -> 'Two'
        30: 30  # 'up arrow' -> 'Up'
    }

    # Adjust annotations based on the manual mapping
    annotation_dir = 'C:\\Users\\Alan\\Desktop\\sc2079\\model24\\labels'  # Path to the labels directory
    adjust_annotations(annotation_dir, old_to_new_mapping)

    # Define new class names
    new_class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bullseye', 'downarrow', 'eight', 'five', 'four', 'leftarrow', 'nine', 'one', 'rightarrow', 'seven', 'six', 'stop', 'three', 'two', 'uparrow']
    
    # Update model configuration to match new number of classes
    model.model.yaml['nc'] = len(new_class_names)  # Set the number of classes
    model.model.yaml['names'] = new_class_names  # Update class names

    print("Model loaded and updated for new class configuration")

    # Start training
    results = model.train(
        data='C:\\Users\\Alan\\Desktop\\sc2079\\model24\\data.yaml',  # Path to your data.yaml
        epochs=80,
        imgsz=640,
        batch=32,
        device=device  # Ensure the training uses the correct device
    )

    # Save the model
    model.save('C:\\Users\\Alan\\Desktop\\sc2079\\model24\\model.pt')

    print("Model training complete and saved")

if __name__ == '__main__':
    main()

