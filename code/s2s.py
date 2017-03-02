import tensorflow as tf

def s2s(model, config, scope, connect):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			model['%s_in0length' %scope] = model['%s_out0length' %connect]
			model['%s_in1length' %scope] = model['%s_out1length' %connect]
			model['%s_in2length' %scope] = model['%s_out1length' %connect]
			model['%s_maxin2length' %scope] = model['%s_maxout2length' %connect]
			model['%s_inputs' %scope] = tf.unstack(model['%s_outputs' %connect], model['%s_maxin2length' %scope], 0, '%s_inputs' %scope)
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = config.getint(scope, 'state_size') if config.has_option(scope, 'state_size') else model['%s_in1length' %scope]
			model['%s_out2length' %scope] = model['%s_in2length' %scope]
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		with tf.variable_scope('initializers'), tf.name_scope('initializers'):
			model['%s_init' %scope] = model['%s_state' %connect]

		with tf.variable_scope('cells'), tf.name_scope('cells'):
			model['%s_rnn' %scope] = getattr(tf.contrib.rnn, config.get(scope, 'cell_type'))(model['%s_out1length' %scope], state_is_tuple = True)

		with tf.variable_scope('stacks'), tf.name_scope('stacks'):
			model['%s_rnns' %scope] = tf.contrib.rnn.MultiRNNCell([model['%s_rnn' %scope]] * config.getint(scope, 'layer_size'))

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if config.get(scope, 'decoder_type') == 'rnn_decoder':
				outputs, state = getattr(tf.contrib.legacy_seq2seq, config.get(scope, 'decoder_type'))(model['%s_inputs' %scope], model['%s_init' %scope], model['%s_rnn' %scope])
			elif config.get(scope, 'decoder_type') == 'attention_decoder':
				outputs, state = getattr(tf.contrib.legacy_seq2seq, config.get(scope, 'decoder_type'))(model['%s_inputs' %scope], model['%s_init' %scope], tf.transpose(model['%s_outputs' %connect], [1, 0, 2]), model['%s_rnn' %scope])
			model['%s_outputs' %scope] = tf.stack(outputs, 0, '%s_outputs' %scope)
			model['%s_state' %scope] = tf.stack(state, 0, '%s_state' %scope)

	return model
