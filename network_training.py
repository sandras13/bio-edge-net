import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
import keras.optimizers
from keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import os
import time
import architectures
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_model_optimization as tfmot

def split_data(X_data, y_data, train_size):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=train_size)
    return X_train, X_val, y_train, y_val

def get_quantized_model(model, X_train):
  def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
      yield [input_value]

  # converter = tf.lite.TFLiteConverter.from_keras_model(model)
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.representative_dataset = representative_data_gen
  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # converter.inference_input_type = tf.uint8
  # converter.inference_output_type = tf.uint8
  # tflite_quant_model = converter.convert()

  # converter = tf.lite.TFLiteConverter.from_keras_model(model)
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.target_spec.supported_types = [tf.float16]
  # tflite_quant_model = converter.convert()

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_quant_model = converter.convert()

  return tflite_quant_model

def get_tflite_interpreter(tflite_quant_model):
  interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
  interpreter.allocate_tensors()

  return interpreter

def get_pruned_model(model, X_data, params):
  prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
  end_step = np.ceil(len(X_data) / params['batch_size']).astype(np.int32) * params['epochs']

  # pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
  #                                                              final_sparsity=0.5,
  #                                                              begin_step=0,
  #                                                              end_step=end_step)}

  pruning_params = {
    'sparsity_m_by_n': (2, 4),
    }

  model_for_pruning = prune_low_magnitude(model, **pruning_params)

  return model_for_pruning

def evaluate_quantized_metrics(interpreter, X_test, y_test):
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  input_shape = input_details['shape']
  predictions = []

  start_time = time.time()
  for i in range(len(X_test)):
      input_data = X_test[i].reshape(input_shape)
      
      if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details['quantization']
        input_data = input_data / input_scale + input_zero_point

      input_data = input_data.astype(input_details['dtype'])
      interpreter.set_tensor(input_details['index'], input_data)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details['index'])
      predictions.append(np.argmax(output_data))

  quantized_time = time.time() - start_time

  y_true_classes = np.argmax(y_test, axis=1)
  accuracy = accuracy_score(y_true_classes, predictions)
  precision = precision_score(y_true_classes, predictions, average='macro')
  recall = recall_score(y_true_classes, predictions, average='macro')
  f1 = f1_score(y_true_classes, predictions, average='macro')

  return accuracy, f1, precision, recall, quantized_time

def train_model(params, X_train, y_train, X_test, y_test):
    model = architectures.ResNet(num_channels=params['num_channels'], num_classes=params['num_classes'],
                                  num_resblocks=params['num_resblocks'], opt=params['resblock_id'])

    def lr_schedule(epoch, lr):
      if epoch % params['step_size'] == 0 and epoch != 0:
        return lr * params['gamma']
      return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)
    optimizer = keras.optimizers.AdamW(learning_rate=params['learning_rate'])

    precision = Precision()
    recall = Recall()

    def f1_metric(y_true, y_pred):
        precision_value = precision(y_true, y_pred)
        recall_value = recall(y_true, y_pred)
        return 2 * ((precision_value * recall_value) / (precision_value + recall_value + 1e-7))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', f1_metric, precision, recall])

    model.fit(X_train, y_train, batch_size=params['batch_size'],
              epochs=params['epochs'], verbose=1, validation_data=(X_test, y_test),
              callbacks=[lr_scheduler])

    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    non_quantized_time = time.time() - start_time

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    return model, accuracy, f1, precision, recall, non_quantized_time

def get_model_size(model_dir):
    total_size =  0
    model_files = {'.pb'}

    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            if os.path.splitext(f)[1] in model_files:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    return total_size


def compare_model_sizes(tflite_quant_model, model):
    with open('converted_quant_model.tflite', 'wb') as f:
      f.write(tflite_quant_model)

    model.save('saved_model', save_format='tf')
    model_dir = 'saved_model'

    non_quantized_model_size = get_model_size(model_dir) / float(2**10)
    quantized_model_size = os.path.getsize('converted_quant_model.tflite') / float(2**10)

    return non_quantized_model_size, quantized_model_size