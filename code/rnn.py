import tensorflow as tf

def rnn(model, config, scope, inseq_len = None, inputs = None):
	sizes = {size: config.getint(scope, size + '_size') for size in ['batch', 'input', 'state', 'layer', 'time', 'class']}
	cell_type, link_type = getattr(tf.nn.rnn_cell, config.get(scope, 'cell_type')), getattr(tf.nn, config.get(scope, 'link_type'))

	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			if inseq_len is None: inseq_len = tf.placeholder(tf.int32, [sizes['batch']], scope + '_inseq_len')
			if inputs is None: inputs = [tf.placeholder(tf.float32, [sizes['batch'], sizes['input']], scope + '_inputs_' + str(_)) for _ in xrange(sizes['time'])]
			model[scope + '_inseq_len'], model[scope + '_inputs'] = inseq_len, inputs

		with tf.variable_scope('initializers'), tf.name_scope('initializers'):
			if config.get(scope, 'cell_type') in ['BasicRNNCell', 'GRUCell']: factor = 1
			elif config.get(scope, 'cell_type') in ['BasicLSTMCell', 'LSTMCell']: factor = 2
			if config.get(scope, 'link_type') == 'rnn':
				model[scope + '_init'] = tf.Variable(tf.truncated_normal([sizes['batch'], factor * sizes['state'] * sizes['layer']]), scope + '_init')
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model[scope + '_init_fw'] = tf.Variable(tf.truncated_normal([sizes['batch'], factor * sizes['state'] * sizes['layer']]), scope + '_init_fw')
				model[scope + '_init_bw'] = tf.Variable(tf.truncated_normal([sizes['batch'], factor * sizes['state'] * sizes['layer']]), scope + '_init_bw')

		with tf.variable_scope('cells'), tf.name_scope('cells'):
			if config.get(scope, 'link_type') == 'rnn':
				model[scope + '_lstm'] = cell_type(sizes['state'])
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model[scope + '_lstm_fw'] = cell_type(sizes['state'])
				model[scope + '_lstm_bw'] = cell_type(sizes['state'])

		with tf.variable_scope('stacks'), tf.name_scope('stacks'):
			if config.get(scope, 'link_type') == 'rnn':
				model[scope + '_lstms'] = tf.nn.rnn_cell.MultiRNNCell([model[scope + '_lstm']] * sizes['layer'])
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model[scope + '_lstms_fw'] = tf.nn.rnn_cell.MultiRNNCell([model[scope + '_lstm_fw']] * sizes['layer'])
				model[scope + '_lstms_bw'] = tf.nn.rnn_cell.MultiRNNCell([model[scope + '_lstm_bw']] * sizes['layer'])

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			if config.get(scope, 'link_type') == 'rnn':
				model[scope + '_outputs'], model[scope + '_state'] = link_type(model[scope + '_lstms'], model[scope + '_inputs'], model[scope + '_init'], tf.float32, model[scope + '_inseq_len'])
			elif config.get(scope, 'link_type') == 'bidirectional_rnn':
				model[scope + '_outputs'], model[scope + '_state_fw'], model[scope + '_state_bw'] = link_type(model[scope + '_lstms_fw'], model[scope + '_lstms_bw'], model[scope + '_inputs'], model[scope + '_init_fw'], model[scope + '_init_bw'], tf.float32, model[scope + '_inseq_len'])

	return model
