import tensorflow as tf

def dia(model, config, scope, connectsegment, connectfeature):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			model['%s_in0length_segment' %scope] = model['%s_out0length' %connectsegment]
			model['%s_in1length_segment' %scope] = model['%s_out1length' %connectsegment]
			model['%s_in2length_segment' %scope] = model['%s_out2length' %connectsegment]
			model['%s_maxin2length_segment' %scope] = model['%s_maxout2length' %connectsegment]
			model['%s_in0length_feature' %scope] = model['%s_out0length' %connectfeature]
			model['%s_in1length_feature' %scope] = model['%s_out1length' %connectfeature]
			model['%s_in2length_feature' %scope] = model['%s_out2length' %connectfeature]
			model['%s_maxin2length_feature' %scope] = model['%s_maxout2length' %connectfeature]
			model['%s_inputs_segment' %scope] = tf.squeeze(model['%s_outputs' %connectsegment], 2, '%s_inputs_segment' %scope)
			model['%s_inputs_feature' %scope] = tf.unstack(tf.transpose(model['%s_outputs' %connectfeature], [1, 0, 2]), name = '%s_inputs_feature' %scope)
			model['%s_out0length' %scope] = model['%s_in0length_feature' %scope]
			model['%s_out1length' %scope] = config.getint('global', 'speaker_size')
			model['%s_out2length' %scope] = tf.stack([config.getint('global', 'speaker_size') for _ in xrange(model['%s_out0length' %scope])])
			model['%s_maxout2length' %scope] = config.getint('global', 'speaker_size')

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_topsegmentvalues' %scope], model['%s_topsegmentindices' %scope] = tf.nn.top_k(tf.transpose(model['%s_inputs_segment' %scope], [1, 0]), config.getint('global', 'speaker_size'))
			model['%s_scores' %scope] = [tf.gather(feature, index) for feature, index in zip(model['%s_inputs_feature' %scope], tf.unstack(model['%s_topsegmentindices' %scope]))]
			model['%s_normalizedscores' %scope]  = [tf.divide(score, tf.norm(score, 2, 1, True)) for score in model['%s_scores' %scope]]
			model['%s_outputs' %scope] = tf.add(0.5, tf.multiply(0.5, tf.stack([tf.matmul(score, score, transpose_b = True) for score in model['%s_normalizedscores' %scope]], name = '%s_outputs' %scope)))

	return model
