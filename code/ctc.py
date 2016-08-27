import tensorflow as tf

def ctc(model, config, scope, connect):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_in0length' %scope] = model['%s_out0length' %connect]
			model['%s_in2length' %scope] = model['%s_maxout2length' %connect]
			model['%s_outputs' %scope] = tf.unpack(model['%s_outputs' %connect], model['%s_in2length' %scope], 0, '%s_outputs' %scope)

		with tf.variable_scope('labels'), tf.name_scope('labels'):
			model['%s_labels_len' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_labels_len' %scope)
			model['%s_labels_ind' %scope] = tf.placeholder(tf.int64, [None, 2], '%s_labels_ind' %scope)
			model['%s_labels_val' %scope] = tf.placeholder(tf.int32, [None], '%s_labels_val' %scope)
			model['%s_labels' %scope] = tf.SparseTensor(model['%s_labels_ind' %scope], model['%s_labels_val' %scope], [model['%s_in2length' %scope], model['%s_in0length' %scope]])

		with tf.variable_scope('loss'), tf.name_scope('loss'):
			model['%s_loss' %scope] = tf.reduce_sum(tf.nn.ctc_loss(model['%s_outputs' %scope], model['%s_labels' %scope], model['%s_labels_len' %scope]), name = '%s_loss' %scope)

	return model
