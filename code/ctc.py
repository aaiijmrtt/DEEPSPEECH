import tensorflow as tf

def ctc(model, config, scope, outputs = None, labels_len = None, labels_ind = None, labels_val = None):
	sizes = {size: config.getint(scope, size + '_size') for size in ['batch', 'output', 'time']}

	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if outputs is None: outputs = [tf.placeholder(tf.float32, [sizes['batch'], sizes['output']], scope + '_outputs_' + str(_)) for _ in xrange(sizes['time'])]
			else: outputs = tf.unpack(outputs, sizes['time'], 0, scope + '_outputs')
			model[scope + '_outputs'] = outputs

		with tf.variable_scope('labels'), tf.name_scope('labels'):
			if labels_len is None: labels_len = tf.placeholder(tf.int32, [sizes['batch']], scope + '_labels_len')
			if labels_ind is None: labels_ind = tf.placeholder(tf.int64, [None, 2], scope + '_labels_ind')
			if labels_val is None: labels_val = tf.placeholder(tf.int32, [None], scope + '_labels_val')
			model[scope + '_labels_len'], model[scope + '_labels_ind'], model[scope + '_labels_val'] = labels_len, labels_ind, labels_val
			model[scope + '_labels'] = tf.SparseTensor(labels_ind, labels_val, [sizes['time'], sizes['batch']])

		with tf.variable_scope('loss'), tf.name_scope('loss'):
			model[scope + '_loss'] = tf.reduce_sum(tf.nn.ctc_loss(model[scope + '_outputs'], model[scope + '_labels'], model[scope + '_labels_len']))

	return model
