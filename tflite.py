import tensorflow as tf

export_dir = "model"
# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile("model.tflite", "wb") as f:
    f.write(tflite_model)
