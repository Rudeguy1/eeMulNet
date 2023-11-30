import os
import random
import shutil
import json

def select_random_files(source_folder, destination_folder):
    file_list = os.listdir(source_folder)

    num_files = int(len(file_list) * 0.2)

    selected_files = random.sample(file_list, num_files)

    os.makedirs(destination_folder, exist_ok=True)

    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)

        destination_path = os.path.join(destination_folder, file_name)

        shutil.move(source_path, destination_path)

    print(f"{num_files} files selected and moved to {destination_folder} folder.")


def generate_testjson(root_dir, output_file):
    catagories = {'Normal': 0,
                  'Abnormal': 1}
    datset_all = {}
    dataset_test = {}
    for category_name in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_name)

        if os.path.isdir(category_path):
            file_list = [f'/{category_name}/{f}' for f in os.listdir(category_path)
                         if os.path.isfile(os.path.join(category_path, f))]
            for file in file_list:
                datset_all[file] = catagories[category_name]

    keys = list(datset_all.keys())
    values = list(datset_all.values())

    for i in range(len(keys)):
        dataset_test[keys[i]] = values[i]
    with open(f'{output_file}/test.json', 'w') as f:
        json.dump(dataset_test, f, indent=4)
generate_testjson(root_dir='',
                  output_file='')
