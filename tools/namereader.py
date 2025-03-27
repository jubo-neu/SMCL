import os


def list_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolders.append(dir_name)
    return subfolders


def save_to_txt(subfolders, output_file):
    with open(output_file, 'w') as f:
        for subfolder in subfolders:
            f.write(subfolder + '\n')


def main():
    folder_path = input("... your path")

    subfolders = list_subfolders(folder_path)

    if subfolders:
        output_file = input("... your path")

        save_to_txt(subfolders, output_file)
        print("save to ...", output_file)
    else:
        print("no folder")


if __name__ == "__main__":
    main()
