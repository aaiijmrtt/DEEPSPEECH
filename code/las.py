import configparser, sys, os
import tensorflow as tf, numpy as np
import rnn, py, s2s, fnn, ce, util

def las(config):
	model = dict()

	model = rnn.rnn(model, config, 'blstm1')
	model = py.py(model, config, 'py1', 'blstm1')
	model = rnn.rnn(model, config, 'blstm2', 'py1')
	model = py.py(model, config, 'py2', 'blstm2')
	model = rnn.rnn(model, config, 'blstm3', 'py2')
	model = s2s.s2s(model, config, 's2s', 'blstm3')
	model = fnn.fnn(model, config, 'map', 's2s')
	model = ce.ce(model, config, 'ce', 'map')

	model['step'] = tf.Variable(0, trainable = False, name = 'step')
	model['lrate'] = tf.train.exponential_decay(config.getfloat('global', 'lrate'), model['step'], config.getint('global', 'dstep'), config.getfloat('global', 'drate'), staircase = False, name = 'lrate')
	model['optim'] = getattr(tf.train, config.get('global', 'optim'))(model['lrate']).minimize(model['ce_loss'], global_step = model['step'], name = 'optim')

	return model

def feed(features, labelslen, labelsind, labelsval, batch_size, time_size):
	feed_dict = {model['blstm1_inputs'][t]: [features[i][:, t] if t < labelslen[i] else np.zeros(features[i][:, labelslen[i] - 1].shape, np.float32) for i in xrange(batch_size)] for t in xrange(time_size)}
	feed_dict.update({model['blstm1_in2length']: [features[i].shape[1] for i in xrange(batch_size)], model['ce_labels_len']: labelslen, model['ce_labels_ind']: labelsind, model['ce_labels_val']: labelsval})
	return feed_dict

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])
	model = las(config)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		util.train(model, sess, config, sys.argv[2], feed)
		print util.test(model, sess, config, sys.argv[2], feed)
		tf.train.Saver().save(sess, os.path.join(config.get('global', 'path'), 'model'))
