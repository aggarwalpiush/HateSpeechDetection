#!/bin/sh
DATAPATH="../processed_data/en/davidson/"

for i in 'random' 'random_POS' 'all' 'dictionary' 'hierarchical'
#for i in 'random'
do
for j in 'camelcasing' 'snakecasing' 'spacing' 'voweldrop' 'random_masking' 'spelling' 'leetspeak' 'mathspeak' 'reversal' 'firstCharacter' 'phonetic' 'charcaterdrop' 'kebabcasing' 'diacritics'
#for j in 'camelcasing'
do
python classification_models_limited.py --train_data "$DATAPATH"davidson_train_data"$i"_"$j"_obfuscated.txt --dev_data "$DATAPATH"davidson_dev_data"$i"_"$j"_obfuscated.txt
done
done
