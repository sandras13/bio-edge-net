import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
import os
import time

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

# def evaluate_quantized_metrics(interpreter, X_test, y_test):
#   input_details = interpreter.get_input_details()
#   output_details = interpreter.get_output_details()

#   input_shape = input_details[0]['shape']
#   accuracy =  0

#   start_time = time.time()
#   for i in range(len(X_test)):
#     input_data = X_test[i].reshape(input_shape)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     if(np.argmax(output_data) == np.argmax(y_test[i])):
#         accuracy +=  1

#   accuracy = accuracy / len(X_test)
#   quantized_time = time.time() - start_time

#   return accuracy, quantized_time

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
    accuracy = np.mean(np.array(predictions) == np.argmax(y_test, axis=1))
    report = classification_report(np.argmax(y_test, axis=1), predictions)

    return accuracy, quantized_time, report
