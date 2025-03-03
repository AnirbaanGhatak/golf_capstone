import os
import shutil

def modify_first_numbers(content, replacements):
    """
    Modifies the first number in each line based on the given mapping.

    :param content: The original content of the file.
    :param replacements: A dictionary where keys are numbers to replace, and values are the new numbers.
    :return: Modified content as a string.
    """
    lines = content.strip().split("\n")  # Split the content into lines
    modified_lines = []

    for line in lines:
        parts = line.split()  # Split each line into parts (numbers/words)
        if parts and parts[0] in replacements:
            parts[0] = str(replacements[parts[0]])  # Replace the first number
        modified_lines.append(" ".join(parts))  # Reassemble the line

    return "\n".join(modified_lines)  # Return the modified content


def process_all_files(source_folder, destination_folder, replacements, source_folder_images, destination_folder_images):
    """
    Loops through all text files in the source folder, modifies them, and moves them to the destination folder.

    :param source_folder: Path of the source folder containing text files.
    :param destination_folder: Path of the destination folder where modified files will be moved.
    :param replacements: Dictionary mapping old numbers to new numbers.
    """
    if not os.path.exists(destination_folder):
        print("directory path incorrect")  # Create destination folder if it doesn't exist

    for file_name in os.listdir(source_folder):
        if file_name.endswith(".txt"):  # Process only .txt files
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)

            # Read the file content
            with open(source_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Modify the content
            modified_content = modify_first_numbers(content, replacements)

            # Write the modified content back to the file
            with open(source_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            # Move the file to the destination folder
            shutil.move(source_path, destination_path)
            print(f"Modified and moved: {file_name}")



    if not os.path.exists(destination_folder_images):
                print("wrong directory path")  # Create destination folder if it doesn't exist

    for file_name_image in os.listdir(source_folder_images):
        if file_name.endswith(".jpg"):  # Process only .txt files
            source_path_image = os.path.join(source_folder, file_name_image)
            destination_path_image = os.path.join(destination_folder_images, file_name_image)

                # Move the file to the destination folder
            shutil.move(source_path_image, destination_path_image)
            print(f"Modified and moved: {file_name_image}")
        

# Example usage
source_folder = "Zips\\new_zips_data\\golf ball-golf club-handle-golf club-head.v3i.yolov11\\train\\labels"  # Change to your source folder path
destination_folder = "Zips\\new_zips_data\\tuningData\\labels\\train"  # Change to your destination folder path

source_folder_images = "Zips\\new_zips_data\\golf ball-golf club-handle-golf club-head.v3i.yolov11\\train\\images"
destination_folder_images = "Zips\\new_zips_data\\tuningData\\images\\train"

# Define replacements: {old_number: new_number}0
replacements = {
    "2": "0",  # Change 0 to 100
    "0": "3"   # Change 1 to 200
    ""
}

process_all_files(source_folder, destination_folder, replacements, destination_folder_images,source_folder_images)