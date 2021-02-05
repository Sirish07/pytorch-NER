import numpy as np
import pandas as pd
import re

def process(s,saveloc):
	words_case = re.findall(r'\w+', s)
	words = re.findall(r'\w+', s.lower())
	labels = ["O"] * len(words)
	f = open(saveloc, "w")
	for k in range(len(words)):
		f.write(words_case[k]+ " " + labels[k] + '\n')
	f.write('\n')

def prediction(text):

	with open(text, 'r') as f:
		lines = f.readlines()
	ans = ""
	for l in lines:
		if len(l) > 2:
			pairs = l.split()
			pred = pairs[-1]
			if "DEL" in pred:
				continue
			if pairs[0] == "<UNK>":
				continue
			ans += pairs[0] + " "
	return ans
