# What is an Recurrent Neural Network (RNN)?
When most people think of Neural Networks (NN), they're actually thinking of Feedforward Neural Networks (FNN) whose general structure involves sending some number of inputs (X) which are then mathematically transformed a number of times equal to however many hidden layers there are. After going through several layers, it spits out some number of outputs (Y). All information travels in a single direction, hence the name, and it requires large ammounts of training data to provide accuracte outputs.

A good example of this is hand-writing interpretation on mail. By having each input be the shade of each specific pixel in the image, then sending it through the multiple hidden layers, the output will be the hand-written address transcribed into digital text.

A RNN works in a similar way, but the hidden layers work slightly differently. Every in a hidden layer node normally takes X inputs and then sends an output to every node in the next layer, so the output only cares about current inputs. A RNN's nodes actually remember the previous output and use it along with the current inputs to create the current output. This sort of memory is highly usedful in predicting trends based on past data, since you want the network to remember previous inputs (e.g., historical weather data).

# Why use a RNN to generate text?
Both FNNS and RNNs can be used to generate text based on training data. A FNN would take a text input, then return what is most likely to follow that text based on the training data. In a sentence, for example, all it takes as input is the most recent word in the sentence, then output the most likely word to follow that. A RNN does the same thing with one minor change. When taking a word input, it remembers the output word, can help with things like keeping a tone withing sentences. For example, positive words (e.g., happy, accomplish, excitement, etc) within a sentence will be remembered and help indicate that all the next outputs should take that into account to sustain that tone. 

# What does this code do?
This project covers building a RNN model to generate text. The idea is to train a model on a large corpus of text, and then have the model generate new, coherent text based on patterns it has learned. In this case we will be using a text containing Shakespeare's Sonnets (source: Project Gutenburg).
