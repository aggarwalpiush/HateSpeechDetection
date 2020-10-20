#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
	input_tsv_file = sys.argv[1]
	df = pd.read_csv(input_tsv_file, sep='\t',  header=None, names = ["a", "b"])
	#df_text_labels = df[['text', 'labels']]
	df = df.fillna(0)

	df["b"]= df["b"].astype(int)
	df["b"] = df["b"].apply(lambda x: 1 if x > 1 else 0)
	print(df.head(100))

	train, testplusdev = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

	train.to_csv('train.tsv', sep='\t', index=False, header=False)
	dev, test = train_test_split(testplusdev, test_size=0.5, random_state=42, shuffle=True)
	dev.to_csv('dev.tsv', sep='\t', index=False, header=False)
	test.to_csv('test.tsv', sep='\t', index=False, header=False)
	zip_file = zipfile.ZipFile('davidson.zip', 'w')
	zip_file.write('train.tsv', compress_type=zipfile.ZIP_DEFLATED)
	zip_file.write('test.tsv', compress_type=zipfile.ZIP_DEFLATED)
	zip_file.write('dev.tsv', compress_type=zipfile.ZIP_DEFLATED)
	zip_file.close()

if __name__ == '__main__':
	main()






