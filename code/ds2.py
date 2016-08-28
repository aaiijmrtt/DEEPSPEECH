import configparser, sys, os
import tensorflow as tf, numpy as np
import fnn, rnn, cnn, ctc, util

def ds2(config):
	model = dict()

	model = cnn.cnn(model, config, 'cnn')
	model = rnn.rnn(model, config, 'rnn', 'cnn')
	model = fnn.fnn(model, config, 'fnn', 'rnn')
	model = ctc.ctc(model, config, 'ctc', 'fnn')

	model['step'] = tf.Variable(0, trainable = False, name = 'step')
	model['lrate'] = tf.train.exponential_decay(config.getfloat('global', 'lrate'), model['step'], config.getint('global', 'dstep'), config.getfloat('global', 'drate'), staircase = False, name = 'lrate')
	model['optim'] = getattr(tf.train, config.get('global', 'optim'))(model['lrate']).minimize(model['ctc_loss'], global_step = model['step'], name = 'optim')

	return model

def feed(features, labelslen, labelsind, labelsval, batch_size, time_size):
	feed_dict = {model['cnn_inputs']: [[features[i][:, t] if t < labelslen[i] else np.zeros(features[i][:, labelslen[i] - 1].shape, np.float32) for i in xrange(batch_size)] for t in xrange(time_size)]}
	feed_dict.update({model['cnn_in2length']: [features[i].shape[1] for i in xrange(batch_size)], model['ctc_labels_len']: labelslen, model['ctc_labels_ind']: labelsind, model['ctc_labels_val']: labelsval})
	return feed_dict

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])
	model = ds2(config)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		util.train(model, sess, config, sys.argv[2], feed)
		print util.test(model, sess, config, sys.argv[2], feed)
		tf.train.Saver().save(sess, os.path.join(config.get('global', 'path'), 'model'))
