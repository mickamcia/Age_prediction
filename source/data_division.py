import os
import random
import shutil
from globals import *

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f'Failed to delete {file_path} : {e}')

def copy_files(source_dir, destination_dir1, destination_dir2, amount_to_copy):
    file_list = os.listdir(source_dir)
    
    for file_name in file_list:
        source_file = os.path.join(source_dir, file_name)
        label = float(file_name.split('_')[0])
        if label < 4 or label > 90:
            continue
        if amount_to_copy < random.random():
            destination_file = os.path.join(destination_dir1, file_name)
        else:
            destination_file = os.path.join(destination_dir2, file_name)
        shutil.copy(source_file, destination_file)

# Example usage

def main():
    amount_to_copy = 0.10
    clear_folder(path_training_set)
    clear_folder(path_validation_set)
    copy_files(path_data_set, path_training_set, path_validation_set, amount_to_copy)


if __name__ == "__main__":
    main()
