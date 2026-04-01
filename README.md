# What is a Recurrent Neural Network (RNN)?
When most people think of Neural Networks (NN), they're actually thinking of Feedforward Neural Networks (FNN) whose general structure involves sending some number of inputs (X) which are then mathematically transformed a number of times equal to however many hidden layers there are. After going through several layers, it spits out some number of outputs (Y). All information travels in a single direction, hence the name, and it requires large amounts of training data to provide accurate outputs.

A good example of this is hand-writing interpretation on mail. By having each input be the shade of each specific pixel in the image, then sending it through the multiple hidden layers, the NN can determine what letters and numbers are written on the envelope. It is also very easy to obtain massive amount of training data, since so much goes through the postal service annually.

An RNN works identically, but with an additional hidden state vector. This vector represents the entire network's memory at any given time step, being calculated as a function of the current input and the previous hidden state. This provides a form of memory across the entire sequence, allowing the network to use earlier inputs as context to help determine the output of the current input.

This sort of memory is highly useful in predicting the next word in a sentence, provided that it's given enough training data on that specific writing style. A FNN would take a paragraph input and only use a fixed window of previous words to determine the next one, but a RNN would take every earlier word into context when determining the next one.

# Why use a RNN to generate text?
Both Feedforward Neural Networks (FNNs) and Recurrent Neural Networks (RNNs) can be used to generate text based on training data. However, RNNs are better suited for this task because they can retain context across longer sequences.

An FNN processes text by taking a fixed-length input (e.g., small window of words) and predicting the most likely word to follow based on training data. However, since it does not have any memory of prior words beyond that limited window, it struggles to generate coherent, contextually consistent text.

In contrast, an RNN remembers past outputs through a hidden state, which evolves as new inputs are processed. This means that when generating text, an RNN can maintain context over an entire sentence or even multiple sentences. For example, if an RNN encounters a positive word like happy, it can influence subsequent word choices to maintain a positive tone. Similarly, it can help preserve grammatical consistency, track subject-verb agreement, and ensure pronoun references remain correct.

This ability to retain and utilize past context makes RNNs highly effective for tasks like text generation, speech recognition, and machine translation, where coherence and long-term dependencies are essential.

# What does this code do?
This project covers building a RNN model to generate text. The idea is to train a model on a large corpus of text, and then have the model generate new, coherent text based on patterns it has learned. In this case we will be using a text containing Shakespeare's Sonnets (source: Project Gutenburg).
    
