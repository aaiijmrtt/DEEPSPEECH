import os
import numpy as np
import mfcc

def prep(config, dirname):
	mfcc_size = config.getint('mfcc', 'n_mfcc')

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.wav'):
				_file = os.path.join(dirpath, filename)
				_mfcc = mfcc.audio2MFCC(_file, config)
				_delta = mfcc.MFCC2delta(_mfcc)
				_augmented = np.vstack([_mfcc, _delta])
				np.savetxt('%s_%i.mfcc' %(_file[: -4], mfcc_size), _augmented)

def dataset(config, dirname, split = [0.8, 0.9], timelimit = 1000):
	n_mfcc, data = config.getint('mfcc', 'n_mfcc'), dict()
	for subdir in os.listdir(dirname):
		if os.path.isdir(os.path.join(dirname, subdir)):
			for speaker in os.listdir(os.path.join(dirname, subdir)):
				if speaker not in data: data[speaker] = list()
				for dirpath, dirnames, filenames in os.walk(os.path.join(dirname, subdir, speaker)):
					for filename in filenames:
						if filename.endswith('%i.mfcc' %n_mfcc):
							if np.loadtxt(os.path.join(dirpath, filename)).shape[1] > timelimit: continue
							data[speaker].append(os.path.join(subdir, speaker, filename))

	labels = dict()
	for label, speaker in enumerate(data):
		labels[speaker] = label

	with open(os.path.join(dirname, 'train'), 'w') as trainfile, open(os.path.join(dirname, 'dev'), 'w') as devfile, open(os.path.join(dirname, 'test'), 'w') as testfile:
		for speaker in data:
			for filename in data[speaker][: int(split[0] * len(data[speaker]))]: trainfile.write('%s\t%i\n' %(filename, labels[speaker]))
			for filename in data[speaker][int(split[0] * len(data[speaker])): int(split[1] * len(data[speaker]))]: devfile.write('%s\t%i\n' %(filename, labels[speaker]))
			for filename in data[speaker][int(split[1] * len(data[speaker])): ]: testfile.write('%s\t%i\n' %(filename, labels[speaker]))

def train(model, sess, config, dirname, feed):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size')
	total_loss, inf_count = 0.0, 0

	with open(os.path.join(dirname, 'train')) as filein:
		for line in filein:
			filename, speaker = line.split('\t')
			label, lenfeatures, lenlabels = int(speaker), len(features), 1
			labelslen.append(lenlabels)
			labelsind.append([lenfeatures, label])
			labelsval.append(1.)
			features.append(np.loadtxt(os.path.join(dirname, filename)))

			if lenfeatures + 1 == batch_size:
				loss, _ = sess.run([model['loss'], model['optim']], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
				if loss == :
				else: total_loss += loss
				features, labelslen, labelsind, labelsval = list(), list(), list(), list()

	return loss

def test(model, sess, config, dirname, feed, devtest = True):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size, mfcc_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size'), config.getint('mfcc', 'n_mfcc')
	total_loss, inf_count = 0.0, 0

	with open(os.path.join(dirname, 'train' if devtest else 'test')) as filein:
		for line in filein:
			filename, speaker = line.split('\t')
			label, lenfeatures, lenlabels = int(speaker), len(features), 1
			labelslen.append(lenlabels)
			labelsind.append([lenfeatures, label])
			labelsval.append(1.)
			features.append(np.loadtxt(os.path.join(dirname, filename)))

			if lenfeatures + 1 == batch_size:
				loss = sess.run(model['loss'], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
				if loss == float('inf'): inf_count += 1
				else: total_loss += loss
				features, labelslen, labelsind, labelsval = list(), list(), list(), list()

	return total_loss, inf_count
