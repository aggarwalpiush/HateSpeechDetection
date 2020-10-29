#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs


def main():
    in_filepath = "/Users/paggarwal/github_repos/HateSpeechDetection/raw_data/en/davidson/labeled_data.csv"
    output_path = "/Users/paggarwal/github_repos/HateSpeechDetection/processed_data/en/davidson/labeled_data.csv"

    with codecs.open(in_filepath, 'r', 'utf-8') as in_obj:
        tweet = []
        label = []
        count = 0
        for i, line in enumerate(in_obj):
            if i == 0:
                continue
            tokens = line.split(',')
            if len(tokens) >= 7:
                tweet.append(','.join(tokens[6:]).rstrip('\r\n').replace('\n',''))
                label.append(float(tokens[2]))
                count += 1
            else:
                tweet[count-1] = str(tweet[count-1]) + ' ' + str(','.join(tokens).rstrip('\r\n').replace('\n',''))


    with codecs.open(output_path, 'w', 'utf-8') as out_obj:
        for i, tw in enumerate(tweet):
            out_obj.write( str(tw.replace("\"", ""))+"∫"+str(label[i])+"\n")


if __name__ == '__main__':
    main()

