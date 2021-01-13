# Running experiments for davidson dataset which should be available in train, test and dev files in tab delimited 
format with statement and labels as two columns. can be downloaded from 
https://drive.google.com/file/d/146sp7wW2CMAEXaxkA7E8ckW7F4AXQRXv/view

# obfucate data example

python obfuscate_data.py --original_data '../processed_data/en/davidson/test.txt'

# after generating obfucated text train shallow learning models

python classification_models.py --train_data '../processed_data/en/davidson/train.txt' --dev_data '../processed_data/en/davidson/dev.txt' --vec_scheme 'fasttext'


# evaluate shallow models on obfuscated test files

python evaluate_model.py --test_path '../processed_data/en/davidson' --model_path ../models

# generate and evaluate deep neural network models 

python network_models.py --train_data '../processed_data/en/davidson/train.txt' --dev_data '../processed_data/en/davidson/dev.txt' --test_path '../processed_data/en/davidson'

# fine-tune bert model and evaluate

python bert_network.py --train_data '../processed_data/en/davidson/train.txt' --dev_data '../processed_data/en/davidson/dev.txt' --test_path '../processed_data/en/davidson'



