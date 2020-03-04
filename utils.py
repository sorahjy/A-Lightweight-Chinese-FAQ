# coding = utf-8

import os, re
from collections import defaultdict


def WriteLine(fout, lst):
    fout.write('\t'.join([str(x) for x in lst]) + '\n')


def IsChsStr(z):
    return re.search('^[\u4e00-\u9fa5]+$', z) is not None


def FreqDict2List(dt):
    return sorted(dt.items(), key=lambda d: d[-1], reverse=True)


def LoadCSV(fn):
    ret = []
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            ret.append(lln)
    return ret


def SaveCSV(csv, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            WriteLine(fout, x)


def LoadDict(fn, func=str):
    dict = {}
    with open(fn, encoding="utf-8") as fin:
        for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
            dict[lv[0]] = func(lv[1])
    return dict


def SaveDict(dict, ofn, output0=True):
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in dict.keys():
            if output0 or dict[k] != 0:
                fout.write(str(k) + "\t" + str(dict[k]) + "\n")


def SaveList(st, ofn):
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in st:
            fout.write(str(k) + "\n")

