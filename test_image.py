import shutil
import os

source_folder = "Zips\\ball_knowledge_zips_ext\\mygolfball.v1i.yolov11\\valid\\labels"
destination_folder = "Zips\\ball_knowledge\\labels\\val"

if not os.path.exists(destination_folder):
        print("wrong directory path")  # Create destination folder if it doesn't exist

for file_name in os.listdir(source_folder):
    if file_name.endswith(".txt"):  # Process only .txt files
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        # Move the file to the destination folder
        shutil.move(source_path, destination_path)
        print(f"Modified and moved: {file_name}")