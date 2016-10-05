# based on audioSegmentation from https://github.com/tyiannak/pyAudioAnalysis

import os, sys
import sklearn, hmmlearn, numpy, scipy, matplotlib.pyplot
import sklearn.discriminant_analysis, sklearn.cluster, hmmlearn.hmm
from scipy.spatial import distance
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO

def trainHMM(features, labels):
	uLabels, nFeatures = numpy.unique(labels), features.shape[0]
	nComps = len(uLabels)
	startprob, transmat, means, cov = numpy.zeros((nComps, )), numpy.zeros((nComps, nComps)), numpy.zeros((nComps, nFeatures)), numpy.zeros((nComps, nFeatures))

	if features.shape[1] < labels.shape[0]:
		print >> sys.stderr, 'trainHMM warning: number of shortterm feature vectors must be >= labels length'
		labels = labels[0: features.shape[1]]

	for i, u in enumerate(uLabels): startprob[i] = numpy.count_nonzero(labels == u)
	startprob = startprob / startprob.sum()
	for i in range(labels.shape[0] - 1): transmat[int(labels[i]), int(labels[i + 1])] += 1
	for i in range(nComps):	transmat[i, :] /= transmat[i, :].sum()
	for i in range(nComps): means[i, :] = numpy.matrix(features[:, numpy.nonzero(labels == uLabels[i])[0]].mean(axis = 1))
	for i in range(nComps): cov[i, :] = numpy.std(features[:, numpy.nonzero(labels == uLabels[i])[0]], axis = 1)

	return startprob, transmat, means, cov

def speakerDiarization(fileName, sRange = xrange(2, 10), mtSize = 2.0, mtStep = 0.2, stWin = 0.05, LDAdim = 35):
	Fs, x = audioBasicIO.readAudioFile(fileName)
	x = audioBasicIO.stereo2mono(x)
	duration = len(x) / Fs

	Classifier1, MEAN1, STD1, classNames1, mtWin1, mtStep1, stWin1, stStep1, computeBEAT1 = aT.loadKNNModel(os.path.join('/home/aaiijmrtt/Code/deepspeech/res/pyAudioAnalysis/data', 'knnSpeakerAll'))
	Classifier2, MEAN2, STD2, classNames2, mtWin2, mtStep2, stWin2, stStep2, computeBEAT2 = aT.loadKNNModel(os.path.join('/home/aaiijmrtt/Code/deepspeech/res/pyAudioAnalysis/data', 'knnSpeakerFemaleMale'))

	MidTermFeatures, ShortTermFeatures = aF.mtFeatureExtraction(x, Fs, mtSize * Fs, mtStep * Fs, round(Fs * stWin), round(Fs * stWin * 0.5))
	MidTermFeatures2 = numpy.zeros((MidTermFeatures.shape[0] + len(classNames1) + len(classNames2), MidTermFeatures.shape[1]))

	for i in range(MidTermFeatures.shape[1]):
		curF1 = (MidTermFeatures[:, i] - MEAN1) / STD1
		curF2 = (MidTermFeatures[:, i] - MEAN2) / STD2

		Result, P1 = aT.classifierWrapper(Classifier1, 'knn', curF1)
		Result, P2 = aT.classifierWrapper(Classifier2, 'knn', curF2)

		MidTermFeatures2[0: MidTermFeatures.shape[0], i] = MidTermFeatures[:, i]
		MidTermFeatures2[MidTermFeatures.shape[0]: MidTermFeatures.shape[0] + len(classNames1), i] = P1 + 0.0001
		MidTermFeatures2[MidTermFeatures.shape[0] + len(classNames1):, i] = P2 + 0.0001

	MidTermFeatures = MidTermFeatures2
	iFeaturesSelect = range(8, 21) + range(41, 54)
	MidTermFeatures = MidTermFeatures[iFeaturesSelect, :]

	MidTermFeaturesNorm, MEAN, STD = aT.normalizeFeatures([MidTermFeatures.T])
	MidTermFeaturesNorm = MidTermFeaturesNorm[0].T
	numOfWindows = MidTermFeatures.shape[1]

	DistancesAll = numpy.sum(distance.squareform(distance.pdist(MidTermFeaturesNorm.T)), axis = 0)
	MDistancesAll = numpy.mean(DistancesAll)
	iNonOutLiers = numpy.nonzero(DistancesAll < 1.2 * MDistancesAll)[0]

	perOutLier = (100.0 * (numOfWindows - iNonOutLiers.shape[0])) / numOfWindows
	MidTermFeaturesNormOr = MidTermFeaturesNorm
	MidTermFeaturesNorm = MidTermFeaturesNorm[:, iNonOutLiers]

	if LDAdim > 0:
		mtWinRatio, mtStepRatio, mtFeaturesToReduce, numOfFeatures, numOfStatistics = int(round(mtSize / stWin)), int(round(stWin / stWin)), list(), len(ShortTermFeatures), 2
		for i in range(numOfStatistics * numOfFeatures): mtFeaturesToReduce.append(list())

		for i in range(numOfFeatures):
			curPos = 0
			N = len(ShortTermFeatures[i])
			while (curPos < N):
				N1, N2 = curPos, curPos + mtWinRatio
				if N2 > N: N2 = N
				curStFeatures = ShortTermFeatures[i][N1: N2]
				mtFeaturesToReduce[i].append(numpy.mean(curStFeatures))
				mtFeaturesToReduce[i + numOfFeatures].append(numpy.std(curStFeatures))
				curPos += mtStepRatio

		mtFeaturesToReduce = numpy.array(mtFeaturesToReduce)
		mtFeaturesToReduce2 = numpy.zeros((mtFeaturesToReduce.shape[0] + len(classNames1) + len(classNames2), mtFeaturesToReduce.shape[1]))
		for i in range(mtFeaturesToReduce.shape[1]):
			curF1 = (mtFeaturesToReduce[:, i] - MEAN1) / STD1
			curF2 = (mtFeaturesToReduce[:, i] - MEAN2) / STD2
			Result, P1 = aT.classifierWrapper(Classifier1, 'knn', curF1)
			Result, P2 = aT.classifierWrapper(Classifier2, 'knn', curF2)
			mtFeaturesToReduce2[0: mtFeaturesToReduce.shape[0], i] = mtFeaturesToReduce[:, i]
			mtFeaturesToReduce2[mtFeaturesToReduce.shape[0]: mtFeaturesToReduce.shape[0] + len(classNames1), i] = P1 + 0.0001
			mtFeaturesToReduce2[mtFeaturesToReduce.shape[0] + len(classNames1):, i] = P2 + 0.0001

		mtFeaturesToReduce = mtFeaturesToReduce2
		mtFeaturesToReduce = mtFeaturesToReduce[iFeaturesSelect, :]
		mtFeaturesToReduce, MEAN, STD = aT.normalizeFeatures([mtFeaturesToReduce.T])
		mtFeaturesToReduce = mtFeaturesToReduce[0].T
	
		Labels = numpy.zeros((mtFeaturesToReduce.shape[1], ))
		LDAstep = 1.0
		LDAstepRatio = LDAstep / stWin

		for i in range(Labels.shape[0]): Labels[i] = int(i * stWin / LDAstepRatio)
		clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components = LDAdim)
		clf.fit(mtFeaturesToReduce.T, Labels)

		MidTermFeaturesNorm = (clf.transform(MidTermFeaturesNorm.T)).T

	clsAll, silAll, centersAll = list(), list(), list()

	for iSpeakers in sRange:
		k_means = sklearn.cluster.KMeans(n_clusters = iSpeakers)
		k_means.fit(MidTermFeaturesNorm.T)
		cls = k_means.labels_
		means = k_means.cluster_centers_

		clsAll.append(cls)
		centersAll.append(means)
		silA, silB = list(), list()
		for c in range(iSpeakers):
			clusterPerCent = numpy.nonzero(cls == c)[0].shape[0] / float(len(cls))
			if clusterPerCent < 0.02:
				silA.append(0.0)
				silB.append(0.0)
			else:
				MidTermFeaturesNormTemp = MidTermFeaturesNorm[:, cls == c]
				Yt = distance.pdist(MidTermFeaturesNormTemp.T)
				silA.append(numpy.mean(Yt) * clusterPerCent)
				silBs = list()
				for c2 in range(iSpeakers):
					if c2 != c:
						clusterPerCent2 = numpy.nonzero(cls == c2)[0].shape[0] / float(len(cls))
						MidTermFeaturesNormTemp2 = MidTermFeaturesNorm[:, cls == c2]
						Yt = distance.cdist(MidTermFeaturesNormTemp.T, MidTermFeaturesNormTemp2.T)
						silBs.append(numpy.mean(Yt) * (clusterPerCent+clusterPerCent2) / 2.0)
				silBs = numpy.array(silBs)
				silB.append(min(silBs))
		silA, silB, sil = numpy.array(silA), numpy.array(silB), list()
		for c in range(iSpeakers): sil.append((silB[c] - silA[c]) / (max(silB[c],  silA[c]) + 0.00001))
		silAll.append(numpy.mean(sil))

	imax = numpy.argmax(silAll)
	nSpeakersFinal = sRange[imax]

	cls = numpy.zeros((numOfWindows, ))
	for i in range(numOfWindows):
		j = numpy.argmin(numpy.abs(i - iNonOutLiers))
		cls[i] = clsAll[imax][j]

	startprob, transmat, means, cov = trainHMM(MidTermFeaturesNormOr, cls)
	hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], 'diag')
	hmm.startprob_ = startprob
	hmm.transmat_ = transmat
	hmm.means_ = means
	hmm.covars_ = cov
	cls = hmm.predict(MidTermFeaturesNormOr.T)
	cls = scipy.signal.medfilt(cls, 13)
	cls = scipy.signal.medfilt(cls, 11)

	sil = silAll[imax]
	classNames = ['SPEAKER{0:d}'.format(c) for c in range(nSpeakersFinal)]

	return cls, classNames, duration, mtStep, silAll

def plot(cls, classNames, duration, mtStep, silAll, sRange, name, filename):
	fig = matplotlib.pyplot.figure()
	fig.suptitle(name, fontsize = 18)

	if len(sRange) == 1: ax1 = fig.add_subplot(111)
	else: ax1 = fig.add_subplot(211)

	ax1.set_yticks(numpy.array(range(len(classNames))))
	ax1.axis((0, duration, -1, len(classNames)))
	ax1.set_yticklabels(classNames)
	ax1.plot(numpy.array(range(len(cls))) * mtStep + mtStep / 2.0, cls)

	matplotlib.pyplot.xlabel('TIME')
	if len(sRange) != 1:
		matplotlib.pyplot.subplot(212)
		matplotlib.pyplot.plot(sRange, silAll)
		matplotlib.pyplot.xlabel('NUMBER OF CLUSTERS')
		matplotlib.pyplot.ylabel('AVERAGE CLUSTERTING SILHOUETTE')

	matplotlib.pyplot.tight_layout(rect = [0, 0.03, 1, 0.95])
	matplotlib.pyplot.show()
	fig.savefig(filename)

if __name__ == '__main__':
	plot(*speakerDiarization('data/SpeakersInTheWild/sample15.wav', [2]), sRange = [2], name = 'GUIDED DIARIZATION', filename = 'report/images/diarization[sitw15][2].jpg')
	plot(*speakerDiarization('data/SpeakersInTheWild/sample15.wav', xrange(2, 10)), sRange = xrange(2, 10), name = 'UNGUIDED DIARIZATION', filename = 'report/images/diarization[sitw15][0].jpg')
	plot(*speakerDiarization('data/SpeakersInTheWild/sample60.wav', [2]), sRange = [2], name = 'GUIDED DIARIZATION', filename = 'report/images/diarization[sitw60][2].jpg')
	plot(*speakerDiarization('data/SpeakersInTheWild/sample60.wav', xrange(2, 10)), sRange = xrange(2, 10), name = 'UNGUIDED DIARIZATION', filename = 'report/images/diarization[sitw60][0].jpg')
