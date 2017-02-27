import tensorflow as tf

def fnn(model, config, scope, connect = None):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			if connect is None:
				model['%s_in0length' %scope] = config.getint('global', 'batch_size')
				model['%s_in1length' %scope] = config.getint('global', 'input_size')
				model['%s_in2length' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_in2length' %scope)
				model['%s_maxin2length' %scope] = config.getint('global', 'time_size')
				model['%s_inputs' %scope] = [tf.placeholder(tf.float32, [model['%s_in0length' %scope], model['%s_in1length' %scope]], '%s_inputs_%i' %(scope, _)) for _ in xrange(model['%s_maxin2length' %scope])]
			else:
				model['%s_in0length' %scope] = model['%s_out0length' %connect]
				model['%s_in1length' %scope] = model['%s_out1length' %connect]
				model['%s_in2length' %scope] = model['%s_out2length' %connect]
				model['%s_maxin2length' %scope] = model['%s_maxout2length' %connect]
				model['%s_inputs' %scope] = tf.unstack(model['%s_outputs' %connect], model['%s_maxin2length' %scope], 0, '%s_inputs' %scope)
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = config.getint(scope, 'output_size') if config.has_option(scope, 'output_size') else model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		with tf.variable_scope('parameters'), tf.name_scope('parameters'):
			for _ in xrange(config.getint(scope, 'layer_size') - 1):
				model['%s_weights%i' %(scope, _)] = tf.Variable(tf.truncated_normal([model['%s_in1length' %scope], model['%s_in1length' %scope]]), name = '%s_weights%i' %(scope, _))
				model['%s_biases%i' %(scope, _)] = tf.Variable(tf.truncated_normal([1, model['%s_in1length' %scope]]), name = '%s_biases%i' %(scope, _))
			model['%s_weights%i' %(scope, config.getint(scope, 'layer_size') - 1)] = tf.Variable(tf.truncated_normal([model['%s_in1length' %scope], model['%s_out1length' %scope]]), name = '%s_weights%i' %(scope, config.getint(scope, 'layer_size') - 1))
			model['%s_biases%i' %(scope, config.getint(scope, 'layer_size') - 1)] = tf.Variable(tf.truncated_normal([1, model['%s_out1length' %scope]]), name = '%s_biases%i' %(scope, config.getint(scope, 'layer_size') - 1))

		def ff(inp):
			for _ in xrange(config.getint(scope, 'layer_size')):
				inp = getattr(tf.nn, config.get(scope, 'active_type'))(tf.add(tf.matmul(inp, model['%s_weights%i' %(scope, _)]), model['%s_biases%i' %(scope, _)]), '%s_feedforward%i' %(scope, _))
			return inp

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_scan' %scope] = tf.map_fn(ff, model['%s_inputs' %scope], tf.float32, name = '%s_scan' %scope)
			model['%s_outputs' %scope] = tf.transpose(model['%s_scan' %scope], [1, 0, 2], '%s_outputs' %scope)

	return model
