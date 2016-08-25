import os
import numpy as np

def prep(config, dirname):
	mfcc_size = config.getint('mfcc', 'n_mfcc')

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.flac'):
				file = os.path.join(dirpath, filename)
				_mfcc = mfcc.audio2MFCC(file, config)
				_delta = mfcc.MFCC2delta(_mfcc)
				_augmented = np.vstack([_mfcc, _delta])
				np.savetxt('%s_%i.mfcc' %(file[: -5], mfcc_size), _augmented)

def train(model, sess, config, dirname, feed):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size')

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.txt'):
				for line in open(os.path.join(dirpath, filename)):
					labels = [ord(c) - 97 if ord(c) > 96 else 26 for c in ' '.join(line.split()[1: ]).lower()[: 3]]
					lenfeatures, lenlabels = len(features), len(labels)
					labelslen.append(lenlabels)
					labelsind.extend([[lenfeatures, i] for i in xrange(lenlabels)])
					labelsval.extend(labels)
					features.append(np.loadtxt(os.path.join(dirpath, '%s_%i.mfcc' %(line.split()[0], config.getint('mfcc', 'n_mfcc'))))[:, : 10])

					if lenfeatures + 1 == batch_size:
						sess.run([model['optim']], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
						features, labelslen, labelsind, labelsval = list(), list(), list(), list()

def test(model, sess, config, dirname, feed):
	features, labelslen, labelsind, labelsval = list(), list(), list(), list()
	batch_size, time_size, mfcc_size = config.getint('global', 'batch_size'), config.getint('global', 'time_size'), config.getint('mfcc', 'n_mfcc')
	ctc_loss, inf_count = 0.0, 0

	for dirpath, dirnames, filenames in os.walk(dirname):
		for filename in filenames:
			if filename.endswith('.txt'):
				for line in open(os.path.join(dirpath, filename)):
					labels = [ord(c) - 97 if ord(c) > 96 else 26 for c in ' '.join(line.split()[1: ]).lower()[: 3]]
					lenfeatures, lenlabels = len(features), len(labels)
					labelslen.append(lenlabels)
					labelsind.extend([[lenfeatures, i] for i in xrange(lenlabels)])
					labelsval.extend(labels)
					features.append(np.loadtxt(os.path.join(dirpath, '%s_%i.mfcc' %(line.split()[0], mfcc_size)))[:, : 10])

					if lenfeatures + 1 == batch_size:
						loss = sess.run(model['ctc_loss'], feed_dict = feed(features, labelslen, labelsind, labelsval, batch_size, time_size))
						if loss == float('inf'): inf_count += 1
						else: ctc_loss += loss
						features, labelslen, labelsind, labelsval = list(), list(), list(), list()

	return ctc_loss, inf_count
