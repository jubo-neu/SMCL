import os
import shutil


def copy_and_rename_images(source_dir, dest_dir, start_index, end_index, step, ii):
    os.makedirs(dest_dir, exist_ok=True)
    index = start_index
    for i in range(start_index, end_index, step):
        if i % 6 == 0:
            continue
        old_filename = os.path.join(source_dir, f"{i:04d}.png")
        new_filename = os.path.join(dest_dir, f"{ii:04d}_" + f"{index:04d}.png")
        shutil.copyfile(old_filename, new_filename)
        index += 1


def main():
    source_dir = "C:/Users/Lenovo/Desktop/Mir-180_Dataset/lamp/For_Synthesis"
    image_folders = os.listdir(source_dir)
    ii = 0
    for folder in image_folders:
        if os.path.isdir(os.path.join(source_dir, folder)):
            image_source_dir = os.path.join(source_dir, folder, "image")
            copy_and_rename_images(image_source_dir, "images", 1, 12, 1, ii)
            copy_and_rename_images(image_source_dir, "180_views", 13, 24, 1, ii)
            copy_and_rename_images(image_source_dir, "mirror_views", 24, 12, -1, ii)
        ii += 1

    current_dir = os.getcwd()

    folders = ['images', '180_views', 'mirror_views']

    for folder in folders:
        folder_path = os.path.join(current_dir, folder)

        count = 0

        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                new_filename = '{:04d}.png'.format(count)

                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                os.rename(old_file_path, new_file_path)

                count += 1


if __name__ == "__main__":
    main()
