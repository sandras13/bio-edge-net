import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import os
import time
import architectures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def split_data(X_data, y_data, train_size):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=train_size)
    return X_train, X_val, y_train, y_val

def get_quantized_model(model):
  converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
  tflite_quant_model = converter.convert()

  return tflite_quant_model

def get_tflite_interpreter(tflite_quant_model):
  interpreter = tensorflow.lite.Interpreter(model_content=tflite_quant_model)
  interpreter.allocate_tensors()

  return interpreter

def evaluate_quantized_metrics(interpreter, X_test, y_test):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_shape = input_details[0]['shape']
  predictions = []

  start_time = time.time()
  for i in range(len(X_test)):
      input_data = X_test[i].reshape(input_shape)
      interpreter.set_tensor(input_details[0]['index'], input_data)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
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

    optimizer = Adam(learning_rate=params['learning_rate'])

    precision = Precision()
    recall = Recall()

    def f1_metric(y_true, y_pred):
        precision_value = precision(y_true, y_pred)
        recall_value = recall(y_true, y_pred)
        return 2 * ((precision_value * recall_value) / (precision_value + recall_value + 1e-7))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', f1_metric, precision, recall])

    model.fit(X_train, y_train, batch_size=params['batch_size'],
              epochs=params['epochs'], verbose=1, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    return model, accuracy, f1, precision, recall

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

    non_quantized_model_size = get_model_size(model_dir) / float(2**20)
    quantized_model_size = os.path.getsize('converted_quant_model.tflite') / float(2**20)

    return non_quantized_model_size, quantized_model_size