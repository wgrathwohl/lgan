import tflib as lib

import numpy as np
import tensorflow as tf
import lipschitz

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None
DONE=False
def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1., lipschitz_constraint=False, l_iters=2):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim), 
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        result = tf.nn.conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, 1, stride, stride],
            padding='SAME',
            data_format='NCHW'
        )

        if lipschitz_constraint:
            filters_mat = tf.reshape(filters, [-1, output_dim])
            k_shape = filters_mat.get_shape().as_list()
            print(k_shape)
            if k_shape[0] < k_shape[1]:
                KtK = tf.matmul(filters_mat, filters_mat, transpose_b=True)
            else:
                KtK = tf.matmul(filters_mat, filters_mat, transpose_a=True)
            print(KtK.get_shape().as_list())
            u = np.random.random((KtK.get_shape().as_list()[0], 1))
            u = u / np.linalg.norm(u)
            print(np.linalg.norm(u))
            u = tf.constant(u, dtype=tf.float32)
            for l_iter in range(l_iters):
                u = tf.matmul(KtK, u)
                u_norm = tf.norm(u)
                u = u / u_norm
            s = tf.sqrt(u_norm)
            print(s.get_shape().as_list())
            global DONE
            if not DONE:
                if k_shape[0] < k_shape[1]:
                    sv, u, v = tf.svd([tf.transpose(filters_mat)], full_matrices=True)
                else:
                    sv, u, v = tf.svd([filters_mat], full_matrices=True)
                msv = sv[0]
                max_singular_value = tf.reduce_max(msv)
                tf.summary.scalar("exact_s", max_singular_value)
                tf.summary.scalar("approx_s", s)
                DONE = True
            in_hw = np.prod(inputs.get_shape().as_list()[2:])
            out_hw = np.prod(result.get_shape().as_list()[2:])
            sfactor = (out_hw ** .5) / (in_hw ** .5) / (filter_size**2)
            result = sfactor * result / s#max_singular_value


        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases, data_format='NCHW')


        return result
