# What is an Recurrent Neural Network (RNN)?
    When most people think of Neural Networks (NN), they're actually thinking of Feedforward Neural Networks (FNN) whose general structure involves sending some number of inputs (X) which are then mathematically transformed a number of times equal to however many hidden layers there are. After going through several layers, it spits out some number of outputs (Y). All information travels in a single direction, hence the name, and it requires large ammounts of training data to provide accuracte outputs.

    A good example of this is hand-writing interpretation on mail. By having each input be the shade of each specific pixel in the image, then sending it through the multiple hidden layers, the output will be the hand-written address transcribed into digital text.

    A RNN works in a similar way, but the hidden layers work slightly differently. Every in a hidden layer node normally takes X inputs and then sends an output to every node in the next layer, so the output only cares about current inputs. A RNN's nodes actually remember the previous output and use it along with the current inputs to create the current output. This sort of memory is highly usedful in predicting trends based on past data, since you want the network to remember previous inputs (e.g., historical weather data).

# Why use a RNN to generate text?
    Both FNNS and RNNs can be used to generate text based on training data, but a FNN would just take a text input and predict the most likely word to follow the input. This will result in gibberish since 

This project covers building a Recurrent Neural Network (RNN) model to generate text. The idea is to train a model on a large corpus of text, and then have the model generate new, coherent text based on patterns it has learned. 