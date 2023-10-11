import numpy as np
from tqdm import tqdm
from scipy import sparse
from summa.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
	with open(path, 'r') as f:
		lines = f.read().split('\n')[0:-1]
	lines = [l.strip() for l in lines]
	return lines

if __name__ == '__main__':
	set_types = ['train', 'valid', 'test']
	corpuses = [load_data('data/body.'+set_type+'.txt') for set_type in set_types]
	fw = [open('data_s/body.'+set_type+'.txt', 'w') for set_type in set_types]
	count = 0
	for corpus in corpuses:
		with tqdm(total=len(corpus), desc=set_types[count], leave=True, unit='sents', unit_scale=True) as pbar:
			for paragraph in corpus:
				simple_lengh = len(paragraph.split(' '))
				simple = paragraph
				# simple = paragraph.replace('phofnewline', '\n')
				if simple_lengh > 100:
					simple = summarize(paragraph, words=80)
				# if simple_lengh > 120:
				# 	simple = summarize(simple, words=80)
				# elif simple_lengh > 100:
				# 	simple = summarize(simple, words=70)
				# simple = simple.replace('\n', 'phofnewline')
				if len(simple.split(' ')) < 20:
					print(simple)
					simple = paragraph
				fw[count].write(simple)
				fw[count].write('\n')
				pbar.update(1)
			fw[count].close()
			count += 1
			
			