import tflib as lib

import numpy as np
import tensorflow as tf
import lipschitz

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None
DONE=False
def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        gain=1.,
        lipschitz_constraint=False,
        l_iters=4,
        l_samples=10
        ):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot' or (initialization == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')
            weight_values = sample((input_dim, output_dim))
        
        elif initialization[0] == 'uniform':
        
            weight_values = np.random.uniform(
                low=-initialization[1],
                high=initialization[1],
                size=(input_dim, output_dim)
            ).astype('float32')

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = lib.param(
            name + '.W',
            weight_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = lib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(name)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if lipschitz_constraint:
            k_shape = weight.get_shape().as_list()
            if k_shape[0] < k_shape[1]:
                KtK = tf.matmul(weight, weight, transpose_b=True)
            else:
                KtK = tf.matmul(weight, weight, transpose_a=True)
            print(KtK.get_shape().as_list())
            u = tf.random_normal((KtK.get_shape().as_list()[0], l_samples))
            u = u / tf.norm(u, axis=1, keep_dims=True)
            for l_iter in range(l_iters):
                u = tf.matmul(KtK, u)
                u_norm = tf.norm(u, axis=1, keep_dims=True)
                u = u / u_norm
            s = tf.reduce_mean(tf.sqrt(u_norm))
            result = result / s
            global DONE
            if not DONE:
                if k_shape[0] < k_shape[1]:
                    sv, u, v = tf.svd([tf.transpose(weight)], full_matrices=True)
                else:
                    sv, u, v = tf.svd([weight], full_matrices=True)
                msv = sv[0]
                max_singular_value = tf.reduce_max(msv)
                tf.summary.scalar("exact_s", max_singular_value)
                tf.summary.scalar("approx_s", s)
                DONE = True

        if biases:
            result = tf.nn.bias_add(
                result,
                lib.param(
                    name + '.b',
                    np.zeros((output_dim,), dtype='float32')
                )
            )

        return result