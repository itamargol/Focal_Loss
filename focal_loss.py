from keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_loss_function(Y_real, Y_hat):
        pt0 = tf.where(tf.equal(Y_real, 0), Y_hat, tf.zeros_like(Y_hat))
        pt1 = tf.where(tf.equal(Y_real, 1), Y_hat, tf.ones_like(Y_hat))
        return -K.mean(alpha * K.pow(1.0 - pt1, gamma) * K.log(pt1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt0, gamma) * K.log(1.0 - pt0+K.epsilon()))
    return focal_loss_function
