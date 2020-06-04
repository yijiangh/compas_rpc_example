import os
import json
import numpy as np
import random

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras

def nn_forward_pass(x, model_file_path):
    model = keras.models.load_model(model_file_path)
    y = model.predict(np.array(x))
    print(y.shape)
    return y

#############################

def get_model():
  # Create a simple model, just for testing
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = keras.Model(inputs, outputs)
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

#############################

def main():
    here = os.path.dirname(__file__)

    # model = get_model()

    # Train the model.
    # test_input = np.random.random((128, 32))
    test_input = [[] for _ in range(128)]
    for i in range(128):
        for j in range(32):
            test_input[i].append(random.random())

    # test_target = np.random.random((128, 1))
    # model.fit(np.array(test_input), test_target)

    model_path = os.path.join(here, 'simple_test_model.h5')
    # model.save(model_path)
    # reconstructed_model = keras.models.load_model('my_h5_model.h5')
    x_input = test_input

    # model_path = os.path.join(here, 'trained_model_0086.h5')
    # # load an example input that I copied from GH
    # x_path = os.path.join(here, 'x_input.json')
    # with open(x_path) as json_file:
    #     data = json.load(json_file)
    # x_input = np.array(data['x'])
    # print(x_input.shape)

    y = nn_forward_pass(x_input, model_path)
    # np.testing.assert_allclose(model.predict(test_input), y)
    print(y.shape)
    print(y)

if __name__ == '__main__':
    main()