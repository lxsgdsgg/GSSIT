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

def word2id(sent, dic):
	id_vec = []
	for word in sent.split(' '):
		if word in dic:
			id_vec.append(dic.index(word))
	return id_vec

def id2word(ids, dic):
	sent = ""
	for index in ids:
		sent += dic[index] + " " 
	return sent

def word2value(sent, dic):
	value_vec = []
	for word in sent:
		value_vec.append(dic.get(word))
	return value_vec

def trunc(sent, value, trunc_word):
	sents = []
	values = []
	sent_len = len(sent)
	start = 0
	while(1):
		if start >= sent_len:
			return sents, values
		if trunc_word not in sent[start:]:
			return sents, values
		end = sent.index(trunc_word, start)
		if sent[start:end]:
			sents.append(sent[start:end])
			values.append(value[start:end])
		start = end + 1

def simplify(counter, corpus, set_type, word_dic):
	simple_corpus = []
	with tqdm(total=len(corpus), desc=set_type, leave=True, unit='sents', unit_scale=True) as pbar:
		for paragraph in corpus:
			if len(paragraph.split(' ')) > 120:
				paragraph_id = word2id(paragraph, word_dic)
				para_vec = counter.transform([paragraph])
				value_vec = dict([(indix, data) for indix, data in zip(para_vec.indices, para_vec.data)])
				paragraph_tfidf = word2value(paragraph_id, value_vec)
				rank = np.argsort(-np.array(paragraph_tfidf))
				# sent_vec, sent_tfidf = trunc(paragraph_id, paragraph_tfidf, newline_id)
				# sum_tfidf = [sum(np.array(value)) for value in sent_tfidf]
				# avg_tfidf = [sum(np.array(value))/len(value) for value in sent_tfidf]
				simple_para_len = 0
				simple_sentid = []
				count = 0
				# sum_rank = np.argsort(-np.array(sum_tfidf))
				# avg_rank = np.argsort(-np.array(avg_tfidf))
				while(1):
					if count >= len(rank):
						simple_corpus.append(paragraph)
						break
					if simple_para_len >= 80:
						simple_sentid.sort()
						simple_para = []
						for i in simple_sentid:
							simple_para.append(i)
							# simple_para.append(newline_id)
						simple_corpus.append(id2word(simple_para, word_dic))
						break
					if rank[count] not in simple_sentid:
						simple_para_len = simple_para_len + 1
						simple_sentid.append(rank[count])
					# if sum_rank[count] not in simple_sentid:
					# 	simple_para_len = simple_para_len + len(sent_vec[sum_rank[count]]) + 1
					# 	simple_sentid.append(sum_rank[count])
					# if avg_rank[count] not in simple_sentid:
					# 	simple_para_len = simple_para_len + len(sent_vec[sum_rank[count]]) + 1
					# 	simple_sentid.append(avg_rank[count])
					count += 1
			else:
				simple_corpus.append(paragraph)
			pbar.update(1)
	return simple_corpus


if __name__ == '__main__':
	project = 'Cross-project'
	set_types = ['train', 'valid', 'test']
	corpuses = [load_data('data/'+ project +'/'+set_type+'_in.txt') for set_type in set_types]
	titles = [load_data('data/'+ project +'/'+set_type+'_out.txt') for set_type in set_types]
	fw_c = [open('data_s/'+ project +'/body.'+set_type+'.txt', 'w') for set_type in set_types]
	fw_t = [open('data_s/'+ project +'/title.'+set_type+'.txt', 'w') for set_type in set_types]

	count = 0
	counter = TfidfVectorizer()
	train_matrix = counter.fit_transform(corpuses[0])
	word_dic = counter.get_feature_names()
	# newline_id = word_dic.index('phofnewline')
	# valid_matrix = transformer.transform(counter.transform(corpuses[1]))
	# test_matrix = transformer.transform(counter.transform(corpuses[2]))


	for corpus, title in zip(corpuses, titles):
		print(set_types[count] + " begin.")
		simple_corpus = simplify(counter, corpus, set_types[count], word_dic)
		for paragraph, t in zip(simple_corpus, title):
			if ('0a0' in paragraph) or ('0a0' in t):
				continue
			if (len(paragraph.split(' ')) < 30) or (len(t.split(' ')) < 5):
				continue
			fw_c[count].write(paragraph+'\n')
			fw_t[count].write(t+'\n')
			# pbar.update(1)
		fw_c[count].close()
		fw_t[count].close()
		print(set_types[count] + " done.")
		count += 1