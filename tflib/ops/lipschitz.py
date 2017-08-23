import tensorflow as tf

def matrix_symmetric(x):
    return (x + tf.transpose(x, [0, 2, 1])) / 2


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])

    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res


@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    """
    Define the gradient for SVD
    References
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers

    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------
    """
    s, u, v = op.outputs
    v_t = tf.transpose(v, [0, 2, 1])
    with tf.name_scope('K'):
        K = get_eigen_K(s, True)
    inner = matrix_symmetric(K * tf.matmul(v_t, grad_v))

    # Create the shape accordingly.
    u_shape = u.get_shape()[1].value
    v_shape = v.get_shape()[1].value

    # Recover the complete S matrices and its gradient
    eye_mat = tf.eye(v_shape, u_shape)
    realS = tf.matmul(tf.reshape(tf.matrix_diag(s), [-1, v_shape]), eye_mat)
    realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])

    real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(grad_s), [-1, v_shape]), eye_mat)
    real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])

    dxdz = tf.matmul(u, tf.matmul(2 * tf.matmul(realS, inner) + real_grad_S, v_t))
    return dxdz


if __name__ == "__main__":
    import numpy as np
    sess = tf.Session()
    iters = 100
    for i in range(iters):
        h = 128#np.random.randint(2, 1000)
        w = 128#np.random.randint(2, 1000)
        M = .001 * tf.constant(np.random.random((h, w)), dtype=tf.float32)
        s_true = tf.reduce_max(tf.svd(M, compute_uv=False))

        if h < w:
            KtK = tf.matmul(M, M, transpose_b=True)
        else:
            KtK = tf.matmul(M, M, transpose_a=True)
        u = tf.nn.l2_normalize(tf.random_normal((KtK.get_shape().as_list()[0], 1)), 1)
        for l_iter in range(4):
            u = tf.matmul(KtK, u)
            u_norm = tf.norm(u, axis=1, keep_dims=True)
            u = u / u_norm

        s_preds = tf.sqrt(u_norm)
        sp_mean = tf.reduce_mean(s_preds)
        Ms = M / sp_mean
        s_Ms = tf.reduce_max(tf.svd(Ms, compute_uv=False))
        st, sp, sm, other = sess.run([s_true, s_preds, sp_mean, s_Ms])
        print(st, sm, (st - sm)/st, sp.max(), sp.min(), other, (h, w))