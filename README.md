# Thor-
Thor Programing Language

Thor is a high-level programming language designed for building and training neural networks. It provides a simple and intuitive syntax for defining complex neural network architectures, as well as a range of built-in functions for common tasks such as data loading, preprocessing, and evaluation. Thor also supports advanced features such as transfer learning, model pruning, and automatic differentiation, making it a powerful tool for both beginners and experienced practitioners.

TODO List
Here are some ideas for features and improvements that we could add to the Thor language:

#Add support for more layer types, such as recurrent neural network (RNN) layers, attention layers, and normalization layers.
#Add support for more optimizers, such as stochastic gradient descent with momentum (SGDM), Adadelta, and RMSprop.
#Add support for more activation functions, such as leaky ReLU, exponential linear unit (ELU), and hyperbolic tangent (tanh).
#Implement a just-in-time (JIT) compiler to improve the performance of Thor programs.
#Add support for distributed training, allowing Thor programs to be trained on multiple GPUs or machines.
#Develop a standard library of pre-trained models and utility functions to make it easier for users to get started with Thor.
#Create a comprehensive user guide and API documentation to help users learn and use the language effectively.
#Implement a package manager to make it easy to install and manage Thor packages and dependencies.
#Develop a testing framework to ensure the correctness and reliability of Thor programs.
#Create a community forum or mailing list to facilitate discussion and collaboration among Thor users and developers.

You can check Readme.md for more info. 

# Thor Language

Thor is a high-level, declarative programming language for building artificial intelligence systems. It provides a simple and intuitive syntax for defining neural networks, training them, and using them for inference.

## Features

* Support for a wide range of layer types, including dense, conv2d, maxpooling2d, LSTM, GRU, and attention layers.
* Support for a variety of optimizers, including SGD, Adam, RMSprop, Adagrad, and Adadelta.
* Support for a variety of activation functions, including ReLU, sigmoid, tanh, and ELU.
* Support for a variety of loss functions, including mean squared error, categorical crossentropy, binary crossentropy, hinge loss, and Huber loss.
* Support for transfer learning, model pruning, and automatic differentiation.

## Syntax

### Neural Networks

A neural network is defined using the `neural_network` keyword followed by a name and a block of layers. Each layer is defined using the `layer` keyword followed by the layer type and its parameters.

Example:
```python
neural_network my_nn {
  layer dense input_dim=10 output_dim=20 activation=relu;
  layer dense input_dim=20 output_dim=2 activation=softmax;
}


Here's an example of how you might implement the code generation function for the neural_network node:


def generate_code_for_neural_network_node(node):
    # Generate code to create a Keras model with the specified layers
    code = 'import tensorflow as tf\n\n'
    code += 'model = tf.keras.Sequential()\n'
    for layer in node.layers:
        if layer.type == 'dense':
            code += f'model.add(tf.keras.layers.Dense(units={layer.output_dim}, activation="{layer.activation}"))\n'
        elif layer.type == 'conv2d':
            code += f'model.add(tf.keras.layers.Conv2D(filters={layer.output_dim}, kernel_size={layer.kernel_size}, activation="{layer.activation}"))\n'
        # Add code for other layer types...
    code += 'model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])\n'
    return code
This function generates Python code that creates a Keras model with the specified layers. It uses the tensorflow library to define the model and its layers.

Once you've implemented all the code generation functions, you can use them to generate Python code for a given AST. Here's an example of how you might do this:


def generate_code(ast):
    code = ''
    for node in ast:
        code += generate_code_for_node(node)
    return code
This function iterates over the nodes in the AST and calls the appropriate code generation function for each node. It concatenates the generated code into a single string and returns it.

# THOR

THOR is a high-level, declarative programming language for building artificial intelligence systems. It provides a simple and intuitive syntax for defining neural networks, training them, and using them for inference.

Getting Started
Here's an example of a simple program in My Language that defines a neural network with two dense layers and trains it on the MNIST dataset:


neural_network my_nn {
  dense input_dim=784 output_dim=128 activation=relu;
  dense input_dim=128 output_dim=10 activation=softmax;
}

loss_function = categorical_crossentropy;
optimizer = adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999);

data_load('mnist.csv');
preprocess(data, labels);

train(my_nn, data, labels, epochs=10, batch_size=32, learning_rate=0.001);
evaluate(my_nn, data, labels);

Language Reference
Neural Networks
A neural network is defined using the neural_network keyword followed by a name and a block of layers. Each layer is defined using the dense, conv2d, or maxpooling2d keyword followed by its parameters.

Loss Functions
A loss function is defined using the loss_function keyword followed by a name. The available loss functions are mean_squared_error and categorical_crossentropy.

Optimizers
An optimizer is defined using the optimizer keyword followed by a name and its parameters. The available optimizers are sgd and adam.

Data Loading and Preprocessing
Data can be loaded using the data_load keyword followed by a file name. The data can be preprocessed using the preprocess keyword followed by the data and labels.

Training and Evaluation
A neural network can be trained using the train keyword followed by the network, data, labels, and training parameters. The network can be evaluated using the evaluate keyword followed by the network, data, and labels.

Standard Library
Thor includes a standard library of functions and classes that can be imported and used in your code. The available modules are:

math: Mathematical functions like sin, cos, and exp.
array: Array manipulation functions like reshape, concatenate, and split.
file: File I/O functions like read_file, write_file, and list_files.

Examples
Here are some examples of programs written in Thor:

Handwritten digit recognition using a convolutional neural network
Image generation using a generative adversarial network
Text generation using a recurrent neural network
Contributing
We welcome contributions from the community! If you'd like to contribute to the development of Thor, please read our contributing guidelines.

License
My Language is released under the MIT License.


This Readme file provides a brief introduction to THor language, explains how to install and use it, and provides examples and documentation for the standard library. It also includes information on how to contribute to the development of Thor and the license under which it is released.
