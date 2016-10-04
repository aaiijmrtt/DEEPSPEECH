import os
import numpy as np, matplotlib.pyplot as plt

def statisticsls(pathname, mfcc):
	means, maxlength = dict(), 0
	for dirpath, dirnames, filenames in os.walk(pathname):
		for filename in filenames:
			if filename.endswith('_%i.mfcc' %mfcc):
				speaker = filename.split('-')[0]
				array = np.split(np.transpose(np.loadtxt(os.path.join(dirpath, filename))), 2, 1)
				if speaker not in means: means[speaker] = [[], []]
				means[speaker][0].append(np.mean(array[0]))
				means[speaker][1].append(np.mean(array[1]))
				maxlength = max(maxlength, array[0].shape[0])
	return [np.mean(means[speaker][0]) for speaker in means], [np.std(means[speaker][0]) for speaker in means], [np.mean(means[speaker][1]) for speaker in means], [np.std(means[speaker][1]) for speaker in means], maxlength

def statisticssitw(pathname, filename, mfcc):
	means, maxlength = dict(), 0
	for line in open(os.path.join(pathname, filename)):
		name, speaker, gender, mic, sess, start, stop, num = line.strip().split()[: 8]
		if num != '1': continue
		if speaker not in means: means[speaker] = [[], []]
		array = np.split(np.transpose(np.loadtxt(os.path.join(pathname, '%s_%i.mfcc' %(name[:-5], mfcc)))), 2, 1)
		means[speaker][0].append(np.mean(array[0]))
		means[speaker][1].append(np.mean(array[1]))
		maxlength = max(maxlength, array[0].shape[0])
	return [np.mean(means[speaker][0]) for speaker in means], [np.std(means[speaker][0]) for speaker in means], [np.mean(means[speaker][1]) for speaker in means], [np.std(means[speaker][1]) for speaker in means], maxlength

def subplot(title, tag, listofXYlabel):
	fig = plt.figure()
	fig.suptitle(title.upper(), fontsize = 18)
	colours = ['r', 'b', 'g', 'c']
	for i, [X, Y, label] in enumerate(listofXYlabel):
		plt.scatter(X, Y, 10, colours[i], alpha = 0.5, label = label)
  	plt.xlabel('MEANS')
	plt.ylabel('DEVIATIONS')
	plt.legend(loc = 'lower left')
	plt.show()
	fig.savefig('report/images/%s[%s].jpg' %(title.lower(), tag))

def plotls(devc, devo, testc, testo):
	devcX1, devcY1, devcX2, devcY2, devcmaxlength = devc
	devoX1, devoY1, devoX2, devoY2, devomaxlength = devo
	testcX1, testcY1, testcX2, testcY2, testcmaxlength = testc
	testoX1, testoY1, testoX2, testoY2, testomaxlength = testo
	print devcmaxlength, devomaxlength, testcmaxlength, testomaxlength
	for X in [devcX1, devcX2, devoX1, devoX2, testcX1, testcX2, testoX1, testoX2]: print np.mean(X), np.std(X), np.max(X), np.min(X)
	subplot('MFCC', 'ls', [[devcX1, devcY1, 'dev clean'], [devoX1, devoY1, 'dev test'], [testcX1, testcY1, 'test clean'], [testoX1, testoY1, 'testother']])
	subplot('DELTA', 'ls', [[devcX2, devcY2, 'dev clean'], [devoX2, devoY2, 'dev test'], [testcX2, testcY2, 'test clean'], [testoX2, testoY2, 'testother']])

def plotsitw(dev, eva):
	devX1, devY1, devX2, devY2, devmaxlength = dev
	evaX1, evaY1, evaX2, evaY2, evamaxlength = eva
	print devmaxlength, evamaxlength
	for X in [devX1, devX2, evaX1, evaX2]: print np.mean(X), np.std(X), np.max(X), np.min(X)
	subplot('MFCC', 'sitw', [[devX1, devY1, 'dev'], [evaX1, evaY1, 'eval']])
	subplot('DELTA', 'sitw', [[devX2, devY2, 'dev'], [evaX2, evaY2, 'eval']])

if __name__ == '__main__':
	plotls(statisticsls('data/LibreSpeech/dev-clean', 50), statisticsls('data/LibreSpeech/dev-other', 50), statisticsls('data/LibreSpeech/test-clean', 50), statisticsls('data/LibreSpeech/test-other', 50))
	plotsitw(statisticssitw('data/SpeakersInTheWild/dev/', 'keys/meta.lst', 50), statisticssitw('data/SpeakersInTheWild/eval/', 'keys/meta.lst', 50))
