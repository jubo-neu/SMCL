import os


def save_folder_names_to_txt(folder_path, output_file):
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            for dir_name in dirs:
                f.write(dir_name + '\n')


folder_path = 'your path'
output_file = 'your path'
save_folder_names_to_txt(folder_path, output_file)

print("save to:", output_file)
