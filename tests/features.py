import matplotlib.pyplot as plt, librosa as lr
from pyAudioAnalysis import audioFeatureExtraction as afe

def subplot(subplot, listofvaluelabel, xlabel, ylabel, yscale = None, legend = None):
	plt.subplot(*subplot)
	for value, label in listofvaluelabel: plt.plot(value, label = label)
	plt.xlabel(xlabel, fontsize = 10)
	plt.ylabel(ylabel, fontsize = 10)
	if yscale is not None: plt.yscale(yscale)
	if legend is not None: plt.legend(loc = 'lower left', prop = {'size': 6})

def plot(signal, sampfreq, features):
	fig = plt.figure('WAVEFORM', figsize = (12, 8), dpi = 100)
	lr.display.waveplot(signal, sampfreq)
	plt.xlabel('TIME')
	plt.ylabel('INTENSITY')
	plt.show()
	fig.savefig('report/images/waveform[%s].jpg' %tag)

	fig = plt.figure('FEATURES', figsize = (12, 8), dpi = 100)
	subplot((3, 2, 1), [[features[0, :], None]], 'FRAME', 'ZERO CROSSING RATE', 'log')
	subplot((3, 2, 2), [[features[1, :], 'ENERGY'], [features[2, :], 'ENTROPY']], 'FRAME', 'ENERGY', 'log', True)
	subplot((3, 2, 3), [[features[3, :], 'CENTROID'], [features[4, :], 'SPREAD'], [features[5, :], 'ENTROPY'], [features[6, :], 'FLUX'],  [features[7, :], 'ROLLOFF']],  'FRAME', 'SPECTRUM', 'log', True)
	subplot((3, 2, 4), [[features[i, :], None] for i in xrange(9, 21)], 'FRAME', 'MFCC')
	subplot((3, 2, 5), [[features[i, :], None] for i in xrange(21, 33)], 'FRAME', 'CHROMA VECTOR', 'log')
	subplot((3, 2, 6), [[features[33, :], None]], 'FRAME', 'CHROMA DEVIATION', 'log')

	plt.tight_layout()
	plt.show()
	fig.savefig('report/images/features[%s].jpg' %tag)

def features(filename, tag):
	signal, sampfreq = lr.load(filename)
	features = afe.stFeatureExtraction(signal, sampfreq, 0.050 * sampfreq, 0.025 * sampfreq)
	return signal, sampfreq, features

if __name__ == '__main__':
	plot(*features('data/LibreSpeech/sample.wav', 'ls'))
	plot(*features('data/SpeakersInTheWild/sample15.wav', 'sitw'))
