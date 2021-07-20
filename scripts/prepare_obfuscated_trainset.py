#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import os
from global_variables import OBFUSCATED_SPAN, OBFUSCATED_STRATEGY
from args import get_args
import shutil

args = get_args()

prefix = os.path.basename(os.path.dirname(args.original_data))
basepath = os.path.dirname(args.original_data)

for sp in OBFUSCATED_SPAN:
    for st in OBFUSCATED_STRATEGY:
        obfuscated_file = os.path.join(basepath, '%s_train_data%s_%s_obfuscated.txt' %(prefix, sp, st))
        if os.path.exists(obfuscated_file):
            with codecs.open(obfuscated_file.replace('.txt','_with_original.txt'), 'w', 'utf-8') as wfd:
                for f in [args.original_data, obfuscated_file]:
                    with codecs.open(f, 'r', 'utf-8') as fd:
                        shutil.copyfileobj(fd, wfd)
            lines_seen = set() # holds lines already seen
            outfile = open(obfuscated_file.replace('.txt','_with_original_deduped.txt'), "w")
            for line in open(obfuscated_file.replace('.txt','_with_original.txt'), "r"):
                if line not in lines_seen: # not a duplicate
                    outfile.write(line)
                    lines_seen.add(line)
            outfile.close()
            os.remove(obfuscated_file.replace('.txt','_with_original.txt'))



