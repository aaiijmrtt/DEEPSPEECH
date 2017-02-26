import tensorflow as tf

def ctc(model, config, scope, connect):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			model['%s_in0length' %scope] = model['%s_out0length' %connect]
			model['%s_in1length' %scope] = model['%s_out1length' %connect]
			model['%s_in2length' %scope] = model['%s_out2length' %connect]
			model['%s_maxin2length' %scope] = model['%s_maxout2length' %connect]
			model['%s_inputs' %scope] = tf.unstack(model['%s_outputs' %connect], model['%s_maxin2length' %scope], 0, '%s_inputs' %scope)
			model['%s_out0length' %scope] = model['%s_in0length' %connect]
			model['%s_out1length' %scope] = model['%s_in1length' %connect]
			model['%s_out2length' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_out2length' %scope)
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %connect]
			model['%s_beamwidth' %scope] = config.getint(scope, 'beam_width') if config.has_option(scope, 'beam_width') else 100
			model['%s_toppaths' %scope] = config.getint(scope, 'top_paths') if config.has_option(scope, 'top_paths') else 1

		with tf.variable_scope('labels'), tf.name_scope('labels'):
			model['%s_labels_len' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_labels_len' %scope)
			model['%s_labels_ind' %scope] = tf.placeholder(tf.int64, [None, 2], '%s_labels_ind' %scope)
			model['%s_labels_val' %scope] = tf.placeholder(tf.int32, [None], '%s_labels_val' %scope)
			model['%s_labels' %scope] = tf.SparseTensor(model['%s_labels_ind' %scope], model['%s_labels_val' %scope], [model['%s_maxin2length' %scope], model['%s_in0length' %scope]])

		with tf.variable_scope('loss'), tf.name_scope('loss'):
			model['%s_loss' %scope] = tf.reduce_sum(tf.nn.ctc_loss(model['%s_labels' %scope], model['%s_inputs' %scope], model['%s_labels_len' %scope]), name = '%s_loss' %scope)

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_output' %scope] = tf.nn.ctc_beam_search_decoder(model['%s_inputs' %scope], model['%s_out2length' %scope], model['%s_beamwidth' %scope], model['%s_toppaths' %scope])

	return model
