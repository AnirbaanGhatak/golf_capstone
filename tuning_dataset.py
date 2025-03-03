import os
import shutil

def modify_first_numbers(content):
    """
    For each line:
      - If first number is '0', remove it from the final file (don't rewrite it).
      - If first number is '1', change it to '0'.
      - Else, leave the line as is.
    """
    lines = content.strip().split("\n")
    modified_lines = []

    for line in lines:
        # Split into tokens/parts
        parts = line.split()
        if not parts:
            # If line is empty after splitting, skip it
            continue

        first_num = parts[0]

        if first_num == "0":
            # Remove this line from the final file entirely
            # => do not add it to modified_lines
            continue

        elif first_num == "1":
            # Change '1' to '0'
            parts[0] = "0"
            modified_lines.append(" ".join(parts))
        else:
            # Keep the line as is
            modified_lines.append(" ".join(parts))

    return "\n".join(modified_lines)

def process_all_files(source_folder, destination_folder):
    """
    Reads .txt files from source_folder, modifies them per the above logic,
    and then moves them to destination_folder.
    """
    if not os.path.exists(destination_folder):
        print(f"[ERROR] Destination folder '{destination_folder}' does not exist.")
        return

    for file_name in os.listdir(source_folder):
        if file_name.endswith(".txt"):
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)

            # Read the file
            with open(source_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Modify the content
            modified_content = modify_first_numbers(content)

            # Write the modified content back
            with open(source_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            # Move the file
            shutil.move(source_path, destination_path)
            print(f"Modified and moved: {file_name}")

# Example usage
source_folder = "Zips\\ball_knowledge_zips_ext\\club-detect.v1i.yolov11\\valid\\labels"
destination_folder = "Zips\\ball_knowledge\\labels\\val"

process_all_files(source_folder, destination_folder)
