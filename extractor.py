# -*- coding: utf-8 -*-
"""
COMP 6970 project4
extract unigram from the ziped txt files
@author Linyuan Zhang
"""
import numpy as np
import os

def data_extractor():
    filepath = "./CASIS-25_Dataset/"
    labels = []
    text = []
    name = []
    for filename in sorted(os.listdir(filepath)):
        path = filepath + filename
        label = filename.split('_')[0]
        labels.append(label)
        openfile = open(str(path), 'rb').read()
        text.append(openfile)
        name.append(filename)

    temp = [0 for i in range(256)]
    char_count = []

    for line in text:
        temp = [0 for i in range(256)]
        for i in range(len(line)):
            temp[line[i]] += 1
        char_count.append(temp[32:127])
    #print(len(char_count[0]))
    char_countArray = np.array(char_count)
    #for char in text:
    #    char_count[ord(char)] += 1
    #for ii in range(100):
    #   print("\n", name[ii], "\n", char_count[ii])

    file = open("newCASIS-25.txt","w+")
    for index in range(len(char_count)):
        file_name = str(name[index])
        char = str(char_count[index])
        file.write(file_name)
        file.write("\n")
        file.write(char)
        file.write("\n")
    file.close()

data_extractor()
