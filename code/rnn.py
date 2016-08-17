import tensorflow as tf

def rnn(model, config, scope, connect = None):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			if connect is None:
				model['%s_in0length' %scope] = config.getint(scope, 'batch_size')
				model['%s_in1length' %scope] = config.getint(scope, 'input_size')
				model['%s_in2length' %scope] = tf.placeholder(tf.int32, [config.getint(scope, 'batch_size')], '%s_in2length' %scope)
				model['%s_inputs' %scope] = [tf.placeholder(tf.float32, [model['%s_in0length' %scope], model['%s_in1length' %scope]], '%s_inputs_%i' %(scope, _)) for _ in xrange(config.getint(scope, 'time_size'))]
			else:
				model['%s_in0length' %scope] = model['%s_out0length' %connect]
				model['%s_in1length' %scope] = model['%s_out1length' %connect]
				model['%s_in2length' %scope] = model['%s_out2length' %connect]
				model['%s_inputs' %scope] = tf.unpack(model['%s_outputs' %connect], config.getint(scope, 'time_size'), 0, '%s_inputs' %scope)

		with tf.variable_scope('initializers'), tf.name_scope('initializers'):
			if config.get(scope, 'cell_type') in ['BasicRNNCell', 'GRUCell']: factor = 1
			elif config.get(scope, 'cell_type') in ['BasicLSTMCell', 'LSTMCell']: factor = 2

			if config.get(scope, 'link_type') == 'rnn':
				model['%s_init' %scope] = tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], factor * config.getint(scope, 'state_size') * config.getint(scope, 'layer_size')]), '%s_init' %scope)
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model['%s_init_fw' %scope] = tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], factor * config.getint(scope, 'state_size') * config.getint(scope, 'layer_size')]), '%s_init_fw' %scope)
				model['%s_init_bw' %scope] = tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], factor * config.getint(scope, 'state_size') * config.getint(scope, 'layer_size')]), '%s_init_bw' %scope)

		with tf.variable_scope('cells'), tf.name_scope('cells'):
			if config.get(scope, 'link_type') == 'rnn':
				model['%s_lstm' %scope] = getattr(tf.nn.rnn_cell, config.get(scope, 'cell_type'))(config.getint(scope, 'state_size'))
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model['%s_lstm_fw' %scope] = getattr(tf.nn.rnn_cell, config.get(scope, 'cell_type'))(config.getint(scope, 'state_size'))
				model['%s_lstm_bw' %scope] = getattr(tf.nn.rnn_cell, config.get(scope, 'cell_type'))(config.getint(scope, 'state_size'))

		with tf.variable_scope('stacks'), tf.name_scope('stacks'):
			if config.get(scope, 'link_type') == 'rnn':
				model['%s_lstms' %scope] = tf.nn.rnn_cell.MultiRNNCell([model['%s_lstm' %scope]] * config.getint(scope, 'layer_size'))
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model['%s_lstms_fw' %scope] = tf.nn.rnn_cell.MultiRNNCell([model['%s_lstm_fw' %scope]] * config.getint(scope, 'layer_size'))
				model['%s_lstms_bw' %scope] = tf.nn.rnn_cell.MultiRNNCell([model['%s_lstm_bw' %scope]] * config.getint(scope, 'layer_size'))

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if config.get(scope, 'link_type') == 'rnn':
				outputs, state = getattr(tf.nn, config.get(scope, 'link_type'))(model['%s_lstms' %scope], model['%s_inputs' %scope], model['%s_init' %scope], tf.float32, model['%s_in2length' %scope])
				model['%s_outputs' %scope] = tf.pack(outputs, 0, '%s_outputs' %scope)
				model['%s_state' %scope] = tf.pack(state, 0, '%s_state' %scope)
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				outputs, state_fw, state_bw = getattr(tf.nn, config.get(scope, 'link_type'))(model['%s_lstms_fw' %scope], model['%s_lstms_bw' %scope], model['%s_inputs' %scope], model['%s_init_fw' %scope], model['%s_init_bw' %scope], tf.float32, model['%s_in2length' %scope])
				model['%s_outputs' %scope] = tf.pack(outputs, 0, '%s_outputs' %scope)
				model['%s_state_fw' %scope] = tf.pack(state_fw, 0, '%s_state_fw' %scope)
				model['%s_state_bw' %scope] = tf.pack(state_bw, 0, '%s_state_bw' %scope)

			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]

	return model
