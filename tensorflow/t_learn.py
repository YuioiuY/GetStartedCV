import tensorflow as tf

 
#Just give below lines parameters
best_weights = 'bottleneck_fc_model.weights.h5'


img_width, img_height = 150, 150
batch_size = 16
nb_img_samples = 8

model = tf.keras.models.load_model('model_keras.keras')

model.load_weights(best_weights)

model.summary()

