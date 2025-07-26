import os
import json

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def process_subfolders(base_dir='.'):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                files = os.listdir(folder_path)
                image_files = [f for f in files if is_image_file(f)]
                output_path = os.path.join(folder_path, 'file.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(image_files, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(image_files)} images to {output_path}")
            except Exception as e:
                print(f"Error processing {folder_path}: {e}")

if __name__ == '__main__':
    process_subfolders()
