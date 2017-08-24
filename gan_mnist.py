import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot


MODE = 'lgan'#'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
TRAIN_DIR = "/u/wgrathwohl/mnist_gan_{}_big_D".format(MODE)
LIPSCHITZ = MODE == 'lgan'

if os.path.exists(TRAIN_DIR):
    print("Deleting existing train dir")
    import shutil

    shutil.rmtree(TRAIN_DIR)
os.makedirs(TRAIN_DIR)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def batch_scale(in1, in2, out1, out2):
    """
    :return: scales outputs so max ||f(x) - f(y)|| / ||x - y|| = 1
    for x in in1 and y in in2
    """
    in_size = np.prod(in1.get_shape().as_list()[1:])
    out_size = np.prod(out1.get_shape().as_list()[1:])
    in_norm = tf.norm(tf.reshape(in1 - in2, [-1, in_size]), axis=1)
    out_norm = tf.norm(tf.reshape(out1 - out2, [-1, out_size]), axis=1)
    ratios = out_norm / in_norm
    ratio = tf.reduce_max(ratios)
    return out1 / ratio, out2 / ratio


def Discriminator(in1, in2):
    nonlin = LeakyReLU
    if tflib.ops.conv2d.format() == 'NCHW':
        in1 = tf.reshape(in1, [-1, 1, 28, 28])
        in2 = tf.reshape(in2, [-1, 1, 28, 28])
    else:
        in1 = tf.reshape(in1, [-1, 28, 28, 1])
        in2 = tf.reshape(in2, [-1, 28, 28, 1])

    out1 = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, in1, stride=1)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, in2, stride=1)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    out1 = lib.ops.conv2d.Conv2D('Discriminator.12', DIM, DIM, 5, in1, stride=2)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.12', DIM, DIM, 5, in2, stride=2)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    out1 = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, in1, stride=1)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, in2, stride=1)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    out1 = lib.ops.conv2d.Conv2D('Discriminator.22', 2*DIM, 2*DIM, 5, in1, stride=1)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.22', 2*DIM, 2*DIM, 5, in2, stride=1)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    out1 = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, in1, stride=2)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, in2, stride=2)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    out1 = lib.ops.conv2d.Conv2D('Discriminator.32', 4*DIM, 4*DIM, 5, in1, stride=2)
    out2 = lib.ops.conv2d.Conv2D('Discriminator.32', 4*DIM, 4*DIM, 5, in2, stride=2)
    out1 = nonlin(out1)
    out2 = nonlin(out2)
    in1, in2 = batch_scale(in1, in2, out1, out2)

    in1 = tf.reshape(in1, [-1, 4*4*4*DIM])
    in2 = tf.reshape(in2, [-1, 4*4*4*DIM])
    out1 = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, in1)
    out2 = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, in2)
    out1, out2 = batch_scale(in1, in2, out1, out2)

    return tf.reshape(out1, [-1]), tf.reshape(out2, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real, disc_fake = Discriminator(real_data, fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp' or MODE == 'lgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    if MODE == 'wgan-gp':
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

    gen_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    )
    gen_gradvars = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
    disc_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    )
    disc_gradvars = disc_opt.compute_gradients(disc_cost, var_list=disc_params)

    for grad, var in disc_gradvars + gen_gradvars:
        tf.summary.histogram(var.name, var)
        tf.summary.histogram(var.name+'_grad', grad)

    disc_train_op = disc_opt.apply_gradients(disc_gradvars)
    gen_train_op = gen_opt.apply_gradients(gen_gradvars)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    )
    gen_gradvars = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
    disc_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    )
    disc_gradvars = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    for grad, var in disc_gradvars + gen_gradvars:
        tf.summary.histogram(var.name, var)
        tf.summary.histogram(var.name+'_grad', grad)

    disc_train_op = disc_opt.apply_gradients(disc_gradvars)
    gen_train_op = gen_opt.apply_gradients(gen_gradvars)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples, feed_dict={real_data: true_dist})
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        '{}/samples_{}.png'.format(TRAIN_DIR, frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Add tensorboard logging
out_norm = tf.abs(disc_real - disc_fake)
data_shape = real_data.get_shape().as_list()
print(disc_real.get_shape().as_list(), disc_fake.get_shape().as_list(), data_shape)
in_norm = tf.norm(tf.reshape(real_data - fake_data, [data_shape[0], -1]), axis=1)
norm_ratio = out_norm / in_norm
mean_lipschitz = tf.reduce_mean(norm_ratio)
max_lipschitz = tf.reduce_max(norm_ratio)
tf.summary.scalar("disc_cost", disc_cost)
tf.summary.histogram("lipschitz_constants", norm_ratio)
tf.summary.scalar("mean_lipschitz", mean_lipschitz)
tf.summary.scalar("max_lipschitz", max_lipschitz)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(TRAIN_DIR)
# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _data = gen.next()
            _ = session.run(gen_train_op, feed_dict={real_data: _data})

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            if i == 0 and iteration % 100 == 0:
                sum_str, _disc_cost, _ = session.run(
                    [summary_op, disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                summary_writer.add_summary(sum_str, iteration)
            else:
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
