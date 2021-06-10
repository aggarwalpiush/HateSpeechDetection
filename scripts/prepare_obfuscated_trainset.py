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
        if os.file.exists(obfuscated_file):
            with codecs.open(obfuscated_file, 'w', 'utf-8') as wfd:
                for f in [args.original_data, obfuscated_file]:
                    with codecs.open(f, 'r', 'utf-8') as fd:
                        shutil.copyfileobj(fd, wfd)




