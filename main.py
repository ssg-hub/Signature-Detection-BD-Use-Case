from utils.preprocessing import make_df, str_way, rearrange_data, sig_classifier

# defining the path and string to find in case a signature is present
path  = '/assets/becode-signature-object-detection/train_xml'
doc_string = 'DLSignature'

# get the list for labels
data, label_0_list, label_1_list = str_way(path, doc_string)

# populate to dataframe
labeldf = make_df(['File','Label'], data)

# save as a csv
labeldf.to_csv(r'/assets/becode-signature-object-detection/labels.csv', index = False)

# rearrange the data into train, validation, test folders
rearrange_data(labeldf, label_0_list, label_1_list)

# now call the notebook that was initially run on google colab
sig_classifier.py