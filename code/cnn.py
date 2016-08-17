import tensorflow as tf

def cnn(model, config, scope, inputs = None):
	sizes = {size: config.getint(scope, size + '_size') for size in ['batch', 'input', 'time', 'layer']}

	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			if inputs is None: inputs = tf.placeholder(tf.float32, [sizes['time'], sizes['batch'], sizes['input']], scope + '_inputs')
			model[scope + '_inputs'] = inputs
			model[scope + '_transform'] = tf.transpose(tf.reshape(model[scope + '_inputs'], [sizes['time'], sizes['batch'], sizes['input'], 1]), [1, 0, 2, 3])

		for _ in xrange(sizes['layer']):
			if _ == 0: model[scope + '_transform' + str(_)] = model[scope + '_transform']
			else: model[scope + '_transform' + str(_)] = model[scope + '_pooling' + str(_ - 1)]

			with tf.variable_scope('filter' + str(_)), tf.name_scope('filter' + str(_)):
				sizes.update({size + str(_): config.getint(scope, size + str(_) + '_size') for size in ['c1length', 'c2length', 'c1step', 'c2step', 'p1length', 'p2length', 'p1step', 'p2step']})
				model[scope + '_filter' + str(_)] = tf.Variable(tf.truncated_normal([sizes['c1length' + str(_)], sizes['c2length' + str(_)], 1, 1]))
				model[scope + '_stride' + str(_)] = [1, sizes['c1step' + str(_)], sizes['c2step' + str(_)], 1]

			with tf.variable_scope('convolution' + str(_)), tf.name_scope('convolution' + str(_)):
				model[scope + '_convolution' + str(_)] = tf.nn.conv2d(model[scope + '_transform' + str(_)], model[scope + '_filter' + str(_)], model[scope + '_stride' + str(_)], 'VALID')
				if config.get(scope, 'pool' + str(_)) == 'max': pool = tf.nn.max_pool
				elif config.get(scope, 'pool' + str(_)) == 'avg': pool = tf.nn.avg_pool
				model[scope + '_pooling' + str(_)] = pool(model[scope + '_convolution' + str(_)], [1, sizes['p1length' + str(_)], sizes['p2length' + str(_)], 1], [1, sizes['p1step' + str(_)], sizes['p2step' + str(_)], 1], 'VALID')

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model[scope + '_outputs'] = tf.transpose(tf.squeeze(model[scope + '_pooling' + str(_)], [3], scope + '_outputs'), [1, 0, 2])

	return model
