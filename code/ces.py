import tensorflow as tf

def ces(model, config, scope, connect):
	with tf.variable_scope(scope), tf.name_scope(scope):
		with tf.variable_scope('inputs'), tf.name_scope('inputs'):
			model['%s_in0length' %scope] = model['%s_out0length' %connect]
			model['%s_in1length' %scope] = model['%s_out1length' %connect]
			model['%s_in2length' %scope] = model['%s_out2length' %connect]
			model['%s_maxin2length' %scope] = model['%s_maxout2length' %connect]
			model['%s_inputs' %scope] = tf.transpose(tf.squeeze(model['%s_outputs' %connect], 2), [1, 0], name = '%s_inputs' %scope)
			model['%s_out0length' %scope] = model['%s_in0length' %scope]
			model['%s_out1length' %scope] = model['%s_in1length' %scope]
			model['%s_out2length' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_out2length' %scope)
			model['%s_maxout2length' %scope] = model['%s_maxin2length' %scope]

		with tf.variable_scope('labels'), tf.name_scope('labels'):
			model['%s_labels_len' %scope] = tf.placeholder(tf.int32, [model['%s_in0length' %scope]], '%s_labels_len' %scope)
			model['%s_labels_ind' %scope] = tf.placeholder(tf.int64, [None, 2], '%s_labels_ind' %scope)
			model['%s_labels_val' %scope] = tf.placeholder(tf.float32, [None], '%s_labels_val' %scope)
			model['%s_labels' %scope] = tf.sparse_to_dense(model['%s_labels_ind' %scope], [model['%s_in0length' %scope], model['%s_maxin2length' %scope]], model['%s_labels_val' %scope], -1, name = '%s_labels' %scope)

		with tf.variable_scope('loss'), tf.name_scope('loss'):
			model['%s_loss' %scope] = tf.reduce_sum(tf.where(tf.less(model['%s_labels' %scope], tf.zeros([model['%s_in0length' %scope], model['%s_maxin2length' %scope]], tf.float32)), tf.zeros([model['%s_in0length' %scope], model['%s_maxin2length' %scope]], tf.float32), -tf.add(tf.multiply(model['%s_labels' %scope], tf.log(tf.add(model['%s_inputs' %scope], 1e-10))), tf.multiply(tf.subtract(1., model['%s_labels' %scope]), tf.log(tf.add(tf.subtract(1., model['%s_inputs' %scope]), 1e-10))))), name = '%s_loss' %scope)

		with tf.variable_scope('outputs'), tf.name_scope('outputs'):
			model['%s_output' %scope] = model['%s_inputs' %scope]

	return model
