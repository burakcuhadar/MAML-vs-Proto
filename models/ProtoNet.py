import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ProtoNet(tf.keras.Model):

	def __init__(self, num_filters, latent_dim):
		super(ProtoNet, self).__init__()
		self.num_filters = num_filters
		self.latent_dim = latent_dim
		num_filter_list = self.num_filters + [latent_dim]
		self.convs = []
		for i, num_filter in enumerate(num_filter_list):
			block_parts = [
				layers.Conv2D(
					filters=num_filter,
					kernel_size=3,
					padding='SAME',
					activation='linear'),
			]

			block_parts += [layers.BatchNormalization()]
			block_parts += [layers.Activation('relu')]
			block_parts += [layers.MaxPool2D()]
			block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
			self.__setattr__("conv%d" % i, block)
			self.convs.append(block)
		self.flatten = tf.keras.layers.Flatten()

	def call(self, inp):
		out = inp
		for conv in self.convs:
			out = conv(out)
		out = self.flatten(out)
		return out

def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
	"""
		calculates the prototype network loss using the latent representation of x
		and the latent representation of the query set
		Args:
			x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
			q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
			labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
			num_classes: number of classes (N) for classification
			num_support: number of examples (S) in the support set
			num_queries: number of examples (Q) in the query set
		Returns:
			ce_loss: the cross entropy loss between the predicted labels and true labels
			acc: the accuracy of classification on the queries
	"""

    # compute the prototypes, shape:[N,D]
	prototypes = tf.stop_gradient(tf.reduce_mean(tf.reshape(x_latent, [num_classes, num_support, -1]), axis=1))

	# shape:[N*Q,N,D]
	repeated_q_latents = tf.keras.backend.repeat(q_latent, num_classes)
	#repeated_q_latents = tf.reshape(repeated_q_latents, [num_classes*num_queries,num_classes,-1])

	# compute the distances from the prototypes, shape:[N*Q, N]
	distances = -tf.norm( repeated_q_latents - prototypes, axis=2 )

	# shape:[N*Q,N]
	labels = tf.stop_gradient(tf.reshape(labels_onehot, [num_classes * num_queries, -1]))

	# compute cross entropy loss
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=distances)
	ce_loss = tf.reduce_sum(loss) / tf.cast(num_queries * num_classes, dtype=tf.float32)

	# compute accuracy
	correct_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(distances, 1), tf.argmax(labels, 1)), dtype=tf.float32))
	acc = correct_count / tf.cast(num_classes * num_queries, dtype=tf.float32)

	return ce_loss, acc

