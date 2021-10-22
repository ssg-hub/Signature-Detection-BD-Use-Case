# import the required libraries
import os
import shutil
import random
import pandas as pd


# using the simplest way using os and string parsing  to get labels from xml files

def make_df(col_list , data_list_of_lists):
    """
    Function to create a dataframe with given 
    column names and list of lists for columns.
    """
    labels_df = pd.DataFrame(data_list_of_lists,
                            columns = col_list
                            ) 

    labels_df.sort_values(by=['File'], inplace=True)

    return labels_df


def str_way(path, doc_string):
    """
    Function to get the lists of filenames and
    corresponding labels, from the given xml file.
    Returns a list of these lists, to be passed
    to create a dataframe, if needed.
    """

    file_list = []
    label_list = [] 
    label_0_list = []
    label_1_list = []

    for root, direc, files in os.walk(path):
        for file in files:
            fname = os.path.join(path,file)
            file_list.append(file) # adding name of file to list
            with open(fname) as myfile:
                fd = myfile.read()
                if doc_string in fd:
                    label_list.append('1')
                    label_1_list.append(file[:-4])
                else:
                    label_list.append('0')
                    label_0_list.append(file[:-4])

    return list(zip(file_list, label_list)), label_0_list, label_1_list




def move_files(basepath, destination, list):
    """
    Function to copy files between folders
    """
    suffix = '.tif'

    for f in list:
        source = os.path.join(basepath, f + suffix)
        shutil.copy(source, destination)



def rearrange_data(labeldf, label_0_list, label_1_list):
    """
    Function to rearrange folders as per the requirements of tensorflow keras.
    Here is the structure that is needed:
    -> model data
        -> test
        -> train
            -> no_sig
            -> sig
        -> val
            -> no_sig
            -> sig
        
    """
    label_0_val_list = random.sample(label_0_list, 82)
    label_1_val_list = random.sample(label_1_list, 124)

    label_0_train_list = set(label_0_list) - set(label_0_val_list)
    label_1_train_list = set(label_1_list) - set(label_1_val_list)

    no_sig = labeldf.loc[labeldf["Label"] == "0"]
    sig = labeldf.loc[labeldf["Label"] == "1"]

    source = 'assets/becode-signature-object-detection/train'

    destination1 = 'assets/becode-signature-object-detection/model_data/train/no_sig'

    destination2 = 'assets/becode-signature-object-detection/model_data/train/sig'

    destination3 = 'assets/becode-signature-object-detection/model_data/val/no_sig'

    destination4 = 'assets/becode-signature-object-detection/model_data/val/sig'

    move_files(source, destination1, label_0_train_list)
    move_files(source, destination2, label_1_train_list)
    move_files(source, destination3, label_0_val_list)
    move_files(source, destination4, label_1_val_list)