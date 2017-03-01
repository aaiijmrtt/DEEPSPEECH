import tensorflow as tf

def rnn(model, config, scope, connect = None):
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
			model['%s_out1length' %scope] = config.getint(scope, 'state_size') if config.has_option(scope, 'state_size') else model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		with tf.variable_scope('cells'), tf.name_scope('cells'):
			if config.get(scope, 'link_type') == 'static_rnn':
				model['%s_rnn' %scope] = getattr(tf.contrib.rnn, config.get(scope, 'cell_type'))(model['%s_out1length' %scope], state_is_tuple = True)
			elif config.get(scope, 'link_type') == 'static_bidirectional_rnn':
				model['%s_rnn_fw' %scope] = getattr(tf.contrib.rnn, config.get(scope, 'cell_type'))(model['%s_out1length' %scope], state_is_tuple = True)
				model['%s_rnn_bw' %scope] = getattr(tf.contrib.rnn, config.get(scope, 'cell_type'))(model['%s_out1length' %scope], state_is_tuple = True)

		with tf.variable_scope('stacks'), tf.name_scope('stacks'):
			if config.get(scope, 'link_type') == 'static_rnn':
				model['%s_rnns' %scope] = tf.contrib.rnn.MultiRNNCell([model['%s_rnn' %scope]] * config.getint(scope, 'layer_size'), state_is_tuple = True)
			elif config.get(scope, 'link_type') == 'static_bidirectional_rnn':
				model['%s_rnns_fw' %scope] = tf.contrib.rnn.MultiRNNCell([model['%s_rnn_fw' %scope]] * config.getint(scope, 'layer_size'), state_is_tuple = True)
				model['%s_rnns_bw' %scope] = tf.contrib.rnn.MultiRNNCell([model['%s_rnn_bw' %scope]] * config.getint(scope, 'layer_size'), state_is_tuple = True)

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if config.get(scope, 'link_type') == 'static_rnn':
				outputs, state = getattr(tf.contrib.rnn, config.get(scope, 'link_type'))(model['%s_rnns' %scope], model['%s_inputs' %scope], dtype = tf.float32, sequence_length = model['%s_in2length' %scope])
				model['%s_outputs' %scope] = tf.stack(outputs, 0, '%s_outputs' %scope)
				model['%s_state' %scope] = state[0]
			elif config.get(scope, 'link_type') == 'static_bidirectional_rnn':
				outputs, state_fw, state_bw = getattr(tf.contrib.rnn, config.get(scope, 'link_type'))(cell_fw = model['%s_rnns_fw' %scope], cell_bw = model['%s_rnns_bw' %scope], inputs = model['%s_inputs' %scope], dtype = tf.float32, sequence_length = model['%s_in2length' %scope])
				model['%s_outputs' %scope] = tf.stack(outputs, 0, '%s_outputs' %scope)
				model['%s_state_c' %scope] = tf.concat([state_fw[0][0], state_bw[0][0]], 1, '%s_state_c' %scope)
				model['%s_state_h' %scope] = tf.concat([state_fw[0][1], state_bw[0][1]], 1, '%s_state_h' %scope)
				model['%s_state' %scope] = (model['%s_state_c' %scope], model['%s_state_h' %scope])

			if config.get(scope, 'link_type') == 'static_bidirectional_rnn':
				model['%s_out1length' %scope] *= 2

	return model
