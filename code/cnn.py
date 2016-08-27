import math, tensorflow as tf

def cnn(model, config, scope, connect = None):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			sizes = {size: config.getint(scope, '%s_size' %size) for size in ['clength', 'cstep', 'plength', 'pstep']}
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
			model['%s_transform' %scope] = tf.transpose(tf.reshape(model['%s_inputs' %scope], [model['%s_maxin2length' %scope], model['%s_in0length' %scope], model['%s_in1length' %scope], 1]), [1, 0, 2, 3], '%s_transform' %scope)
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		for _ in xrange(config.getint(scope, 'layer_size')):
			if _ == 0: model['%s_transform%i' %(scope, _)] = model['%s_transform' %scope]
			else: model['%s_transform%i' %(scope, _)] = model['%s_pooling%i' %(scope, _ - 1)]

			with tf.variable_scope('filter%i' %_), tf.name_scope('filter%s' %_):
				model['%s_filter%i' %(scope, _)] = tf.Variable(tf.truncated_normal([sizes['clength'], sizes['clength'], 1, 1]))
				model['%s_stride%i' %(scope, _)] = [1, sizes['cstep'], sizes['cstep'], 1]

			with tf.variable_scope('convolution%i' %_), tf.name_scope('convolution%i' %_):
				model['%s_convolution%i' %(scope, _)] = tf.nn.conv2d(model['%s_transform%i' %(scope, _)], model['%s_filter%i' %(scope, _)], model['%s_stride%i' %(scope, _)], 'VALID')
				model['%s_out1length' %scope] = int(math.ceil(float(model['%s_out1length' %scope] - sizes['clength'] + 1) / float(sizes['cstep'])))
				model['%s_out2length' %scope] = tf.to_int32(tf.ceil(tf.div(tf.to_float(tf.sub(model['%s_out2length' %scope], sizes['clength'] - 1)), tf.to_float(sizes['cstep']))))
				model['%s_maxout2length' %scope] = int(math.ceil(float(model['%s_maxout2length' %scope] - sizes['clength'] + 1) / float(sizes['cstep'])))
				model['%s_pooling%i' %(scope, _)] = getattr(tf.nn, '%s_pool' %config.get(scope, 'pool'))(model['%s_convolution%i' %(scope, _)], [1, sizes['plength'], sizes['plength'], 1], [1, sizes['pstep'], sizes['pstep'], 1], 'VALID')
				model['%s_out1length' %scope] = int(math.ceil(float(model['%s_out1length' %scope] - sizes['plength'] + 1) / float(sizes['pstep'])))
				model['%s_out2length' %scope] = tf.to_int32(tf.ceil(tf.div(tf.to_float(tf.sub(model['%s_out2length' %scope], sizes['plength'] - 1)), tf.to_float(sizes['pstep']))))
				model['%s_maxout2length' %scope] = int(math.ceil(float(model['%s_maxout2length' %scope] - sizes['plength'] + 1) / float(sizes['pstep'])))

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_outputs' %scope] = tf.transpose(tf.squeeze(model['%s_pooling%i' %(scope, _)], [3], '%s_outputs' %scope), [1, 0, 2])

	return model
