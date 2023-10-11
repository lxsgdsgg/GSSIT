import tqdm
import json

def load_data(path):
	with open(path, 'r') as f:
		lines = f.read().split('\n')[0:-1]
	lines = [l.strip() for l in lines]
	return lines


def filter_newdata():
	# Within-project, Cross-project
	project = 'Within-project'
	set_types = ['train', 'valid', 'test']
	corpuses = [load_data('data_s/'+ project +'/body.'+set_type+'.txt') for set_type in set_types]
	titles = [load_data('data_s/'+ project +'/title.'+set_type+'.txt') for set_type in set_types]
	fw_c = [open('data_s/'+ project +'/body.'+set_type+'.fine.txt', 'w') for set_type in set_types]
	fw_t = [open('data_s/'+ project +'/title.'+set_type+'.fine.txt', 'w') for set_type in set_types]

	count = 0

	for corpus, title in zip(corpuses, titles):
		# with tqdm(total=len(corpus), desc=set_types[count], leave=True, unit='sents', unit_scale=True) as pbar:
		for para, t in zip(corpus, title):
			if ('000' in para) or ('000' in t):
				continue
			if (len(para.split(' ')) < 30) or (len(t.split(' ')) < 5):
				continue
			fw_c[count].write(para+'\n')
			fw_t[count].write(t+'\n')
			# pbar.update(1)
		fw_c[count].close()
		fw_t[count].close()
		count += 1

def filter_data():
	# Within-project, Cross-project
	project = 'Within-project'
	set_types = ['train', 'valid', 'test']
	# with open('data/'+ project +'/'+set_types[0]+'.json', 'r') as fr:
	# 	data = json.load(fr)
	# 	print(data)
	corpuses = [json.load(open('data/'+ project +'/'+set_type+'.json', 'r'))for set_type in set_types]
	print(corpuses[0][0]['Title'])


if __name__ == '__main__':
	# filter_newdata()
	filter_data()


