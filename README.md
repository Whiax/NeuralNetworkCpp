# NeuralNetworkCpp
Basic deep learning framework in C++ / new ShakingTree optimization algorithm

All information can be found here:
https://medium.com/@hyugen.ai/neural-network-in-c-from-scratch-and-backprop-free-optimizer-d2a34bb92688

/!\ The implementation is very basic and mostly made to learn, play and try things with deep neural networks. There's no convolutional networks, no transformers, no Adam etc..
For a professional deep learning library in C++ I'll recommend http://dlib.net/ or the C++ API of Pytorch and Tensorflow.

PS: I worked on this project with Visual Studio / Windows and it seems that some indentations were incorrectly encoded, sorry about that

## Framework Design
### 1. Folders (latest version)
- Neural: Contains classes related to the neural network
- Dataset: Contains one simple dataset class
- Optimizer: Contains a generic optimizer class, a classical backpropagation optimizer and an exotic optimizer
- Misc: Contains activation functions and their derivatives

### 2. Neural Classes
- NeuralNetwork.h/.cpp: the main class containing a model. A NeuralNetwork is composed of multiple layers
- Layer.h/.cpp: A Layer is composed of multiple neurons. It has a type (INPUT/STANDARD/OUTPUT) because it doesn’t work the same way depending on its type and it has an activation function
- Neuron.h/.cpp: A Neuron has a parent layer. A neuron accumulates the input of the neurons connected to it (_accumulated), it outputs that input to its edges after processing it with its _activation_function (which you can change on a per-neuron basis). A neuron knows its _next and _previous edges.
- Edge.h/.cpp: An Edge knows its next neuron (_n), the neuron its coming from (_nb), it has a weight (_w), it can memorize how much its weight was shift (_last_shift) last time it was, and it has a backpropagation memory so that it can retain a part of the chain rule to avoid useless computations

### 3. Dataset / Optimizer
- Dataset.h/.cpp: It gives the data. One input/output is just a vector (again, no matrices, no images). So the list of all inputs is a vector of vector. The Dataset class is also responsible for doing the train/test split.
- Optimizer.h/.cpp: Abstract class that links the network and the dataset together. Child classes must implement a “minimize” method that is called during the optimization process to minimize the loss.
- Backpropagation.h/.cpp: Implementation of the backpropagation optimizer (though the backpropagation computations are in the neuron class for simplification). This class is mostly there to call these methods in the neuron class and to apply the new computed weights. It uses a batch size, the learning rate is a global variable because it’s common to all implemented optimizers and it’s easier to access it from anywhere.
- ShakingTree.h/.cpp: Exotic optimizer that’ll be detailed later.

## Computation example
The algorithm was also made to exactly reproduce the article from Matt Mazur on his website: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/  
Just use the “mattmazur_step_by_step” branch on git, this branch serves only that purpose and contains older code than the code on “master”.

