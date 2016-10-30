import tensorflow as tf

def py(model, config, scope, connect = None):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			if connect is None:
				model['%s_in0length' %scope] = config.getint('global', 'batch_size')
				model['%s_in1length' %scope] = config.getint('global', 'input_size')
				model['%s_in2length' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_in2length' %scope)
				model['%s_maxin2length' %scope] = config.getint('global', 'time_size')
				model['%s_inputs' %scope] = tf.placeholder(tf.float32, [model['%s_maxin2length' %scope], model['%s_in0length' %scope], model['%s_in1length' %scope]], '%s_inputs' %scope)
			else:
				model['%s_in0length' %scope] = model['%s_out0length' %connect]
				model['%s_in1length' %scope] = model['%s_out1length' %connect]
				model['%s_in2length' %scope] = model['%s_out2length' %connect]
				model['%s_maxin2length' %scope] = model['%s_maxout2length' %connect]
				model['%s_inputs' %scope] = model['%s_outputs' %connect]
			model['%s_factor' %scope] = config.getint(scope, 'factor')
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = model['%s_in1length' %scope] * model['%s_factor' %scope]
			model['%s_out2length' %scope] = tf.div(tf.sub(model['%s_in2length' %scope], tf.mod(model['%s_in2length' %scope], model['%s_factor' %scope])), model['%s_factor' %scope])
			model['%s_maxout2length' %scope] = (model['%s_maxin2length' %scope] - model['%s_maxin2length' %scope] % model['%s_factor' %scope]) / model['%s_factor' %scope]

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_transpose' %scope] = tf.transpose(model['%s_inputs' %scope], [0, 2, 1], '%s_transpose' %scope)
			model['%s_transform' %scope] = tf.reshape(model['%s_transpose' %scope], [model['%s_maxout2length' %scope], model['%s_out1length' %scope], model['%s_out0length' %scope]], '%s_transform' %scope)
			model['%s_outputs' %scope] = tf.transpose(model['%s_transform' %scope], [0, 2, 1], '%s_outputs' %scope)

	return model
