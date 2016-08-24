import tensorflow as tf

def s2s(model, config, scope, connect):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			model['%s_in0length' %scope] = config.getint(scope, 'batch_size')
			model['%s_in1length' %scope] = config.getint(scope, 'input_size')
			model['%s_in2length' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_in2length' %scope)
			model['%s_maxin2length' %scope] = config.getint(scope, 'time_size')
			model['%s_inputs' %scope] = [tf.placeholder(tf.float32, [model['%s_in0length' %scope], model['%s_in1length' %scope]], '%s_inputs_%i' %(scope, _)) for _ in xrange(model['%s_maxin2length' %scope])]
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = config.getint(scope, 'state_size') if config.has_option(scope, 'state_size') else model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		with tf.variable_scope('initializers'), tf.name_scope('initializers'):
			if config.get(scope, 'cell_type') in ['BasicRNNCell', 'GRUCell']:
				model['%s_init' %scope] = tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], model['%s_out1length' %scope] * config.getint(scope, 'layer_size')]), '%s_init' %scope)
				model['%s_state' %scope] = model['%s_state' %connect] if '%s_state' %connect in model else tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], model['%s_out1length' %scope] * config.getint(scope, 'layer_size')]), '%s_state' %scope)
			elif config.get(scope, 'cell_type') in ['BasicLSTMCell', 'LSTMCell']:
				model['%s_init' %scope] = tf.Variable(tf.truncated_normal([model['%s_in0length' %scope], 2 * model['%s_out1length' %scope] * config.getint(scope, 'layer_size')]), '%s_init' %scope)
				model['%s_state' %scope] = model['%s_state' %connect] if '%s_state' %connect in model else tf.Variable(tf.truncated_normal([2 * model['%s_out1length' %scope] * config.getint(scope, 'layer_size')]), '%s_state' %scope)

		with tf.variable_scope('cells'), tf.name_scope('cells'):
			model['%s_rnn' %scope] = getattr(tf.nn.rnn_cell, config.get(scope, 'cell_type'))(model['%s_out1length' %scope])

		with tf.variable_scope('stacks'), tf.name_scope('stacks'):
			model['%s_rnns' %scope] = tf.nn.rnn_cell.MultiRNNCell([model['%s_rnn' %scope]] * config.getint(scope, 'layer_size'))

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if config.get(scope, 'decoder_type') == 'rnn_decoder':
				outputs, state = getattr(tf.nn.seq2seq, config.get(scope, 'decoder_type'))(model['%s_inputs' %scope], model['%s_state' %scope], model['%s_rnns' %scope])
			elif config.get(scope, 'decoder_type') == 'attention_decoder':
				outputs, state = getattr(tf.nn.seq2seq, config.get(scope, 'decoder_type'))(model['%s_inputs' %scope], model['%s_state' %scope], model['%s_outputs' %connect], model['%s_rnns' %scope])
			model['%s_outputs' %scope] = tf.pack(outputs, 0, '%s_outputs' %scope)
			model['%s_state' %scope] = tf.pack(state, 0, '%s_state' %scope)

	return model
