import random

def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

class Sentence(object):
	"""docstring for Sentence"""
	def __init__(self, sent, label):
		super(Sentence, self).__init__()
		self.sent = sent
		self.label = label
		
def output_data(datas, type):
	# labels = ['aspect evaluation', 'feature request', 'information giving', 'information seeking', 'problem discovery', 'solution proposal', 'others']
	labels = ['description', 'expected', 'others', 'reproduce']
	data = []
	for i in range(4):
		for temp in datas[i]:
			data.append(Sentence(temp, labels[i]))
	random.shuffle(data)
	x_file = open('data/new/'+ type + "_x.txt", 'w')
	y_file = open('data/new/'+ type + "_y.txt", 'w')
	for sample in data:
		if(len(sample.sent.split(' '))) > 200:
			continue
		x_file.write(sample.sent)
		y_file.write(sample.label)
		x_file.write('\n')
		y_file.write('\n')

def main():
	# files = ['aspect evaluation', 'feature request', 'information giving', 'information seeking', 'problem discovery', 'solution proposal', 'others']
	files = ['des.txt', 'exp.txt', 'oth.txt', 'rep.txt']
	paths = ['data/new/' + file for file in files]
	sentences = [load_data(path) for path in paths]
	lenghs = [len(sentence) for sentence in sentences]
	train_lengh = [round(0.8 * lengh) for lengh in lenghs]
	valid_lengh = [round(0.9 * lengh) for lengh in lenghs]
	train_datas = [sentences[i][:train_lengh[i]] for i in range(4)]
	valid_datas = [sentences[i][train_lengh[i]:valid_lengh[i]] for i in range(4)]
	test_datas = [sentences[i][valid_lengh[i]:] for i in range(4)]
	output_data(train_datas, 'train')
	output_data(valid_datas, 'valid')
	output_data(test_datas, 'test')


if __name__ == '__main__':
	main()
