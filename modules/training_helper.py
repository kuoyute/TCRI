import tensorflow as tf

def calculate_metric_dict(model, dataset):
    mae = tf.constant([0.])
    mse = tf.constant([0.])
    num = 0.

    for image_sequences, labels, feature, frame_ID_ascii, dV in dataset:
        pred = model(image_sequences, feature, training=False)
        
        sample_weight = tf.math.tanh((dV-20)/10)*1000 +1000.1
        
        mae_each = tf.math.reduce_mean((tf.abs(labels-pred))*sample_weight)
        mse_each = tf.math.reduce_mean((tf.abs(labels-pred)**2)*sample_weight)
        
        num+=1
        mae = tf.add(mae, mae_each)
        mse = tf.add(mse, mse_each)
        
    
    MAE = tf.reduce_mean(tf.math.divide(mae, num))
    MSE = tf.reduce_mean(tf.math.divide(mse, num))

    return dict(
        MAE=MAE,
        MSE=MSE
    )
