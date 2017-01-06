import os
import numpy as np

def prep(config, dirname):
	mfcc_size = config.getint('mfcc', 'n_mfcc')

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.flac'):
				_file = os.path.join(dirpath, filename)
				_mfcc = mfcc.audio2MFCC(_file, config)
				_delta = mfcc.MFCC2delta(_mfcc)
				_augmented = np.vstack([_mfcc, _delta])
				np.savetxt('%s_%i.mfcc' %(_file[: -5], mfcc_size), _augmented)

def filtertext(char):
	return char.isalpha() or char.isspace()

def maptext(char):
	if char.isupper(): return ord(char) - 65
	if char.islower(): return ord(char) - 97
	if char.isspace(): return 26
	return -1

def train(model, sess, config, dirname, feed):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size')

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.txt'):
				for line in open(os.path.join(dirpath, filename)):
					labels = map(maptext, filter(filtertext, line.split(' ', 1)[1]))
					lenfeatures, lenlabels = len(features), len(labels)
					labelslen.append(lenlabels)
					labelsind.extend([[lenfeatures, i] for i in xrange(lenlabels)])
					labelsval.extend(labels)
					features.append(np.loadtxt(os.path.join(dirpath, '%s_%i.mfcc' %(line.split()[0], config.getint('mfcc', 'n_mfcc')))))

					if lenfeatures + 1 == batch_size:
						sess.run([model['optim']], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
						features, labelslen, labelsind, labelsval = list(), list(), list(), list()

def test(model, sess, config, dirname, feed):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size, mfcc_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size'), config.getint('mfcc', 'n_mfcc')
	total_loss, inf_count = 0.0, 0

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.txt'):
				for line in open(os.path.join(dirpath, filename)):
					labels = map(maptext, filter(filtertext, line.split(' ', 1)[1]))
					lenfeatures, lenlabels = len(features), len(labels)
					labelslen.append(lenlabels)
					labelsind.extend([[lenfeatures, i] for i in xrange(lenlabels)])
					labelsval.extend(labels)
					features.append(np.loadtxt(os.path.join(dirpath, '%s_%i.mfcc' %(line.split()[0], mfcc_size))))

					if lenfeatures + 1 == batch_size:
						loss = sess.run(model['loss'], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
						if loss == float('inf'): inf_count += 1
						else: total_loss += loss
						features, labelslen, labelsind, labelsval = list(), list(), list(), list()

	return total_loss, inf_count
