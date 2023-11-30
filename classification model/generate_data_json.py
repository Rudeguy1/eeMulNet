import os
import json
from sklearn.model_selection import KFold
# Read training folder data and perform 5 fold cross validation partitioning


def generate_dataset_json(root_dir, output_file):
    """
    under the main folder is the category folder.which saves the corresponding
    image addresses and categories in the json file for each categoty
    :param root_dir:
    :param output_file:
    :return:
    """
    # catagories = {'Normal': 0,
    #               'PDAC': 1,
    #               'NonePDAC': 2}
    catagories = {'Normal': 0,
                  'PDAC': 1}
    kf = KFold(n_splits=5, shuffle=True, random_state=226)
    datset_all = {}

    # traverse all folders under the root directory
    for category_name in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_name)

        if os.path.isdir(category_path):
            file_list = [f'/{category_name}/{f}' for f in os.listdir(category_path)
                         if os.path.isfile(os.path.join(category_path, f))]
            for file in file_list:
                datset_all[file] = catagories[category_name]
    for k, (train_indexs, test_indexs) in enumerate(kf.split(datset_all)):
        keys =list(datset_all.keys())
        values = list(datset_all.values())
        dataset_train = {}
        dataset_test = {}

        for train_index in train_indexs:
            dataset_train[keys[train_index]] = values[train_index]
        for test_index in test_indexs:
            dataset_test[keys[test_index]] = values[test_index]
        with open(f'{output_file}/{k}_train.json', 'w') as f:
            json.dump(dataset_train, f, indent=4)
        with open(f'{output_file}/{k}_test.json', 'w') as f:
            json.dump(dataset_test, f, indent=4)


def generate_dataset_json_imgLabelClassification(root_dir, output_file):
    """
    Under the main folder, there are category folders and corresponding label images for each category.
     The corresponding image addresses, labels, and categories in each category are saved in a JSON file
     and divided into five fold cross sections
    :param root_dir:
    :param output_file:
    :return:
    """
# root_dir = '/home/zhanggf/code/pythonProject_exercise/data/2023_test'
root_dir = '/home/zhanggf/code/Data_seg/Classification/Train_allMoredisease2019'
# output_file = f'/home/zhanggf/code/pythonProject_exercise/data/allAug_pad_gray_crop_compresion_12812864/trainjson.json'
generate_dataset_json(root_dir=root_dir, output_file=root_dir)

