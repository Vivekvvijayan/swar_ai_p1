### About the Classifier
The classifier is a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) trained on over 10,000 hours of Carnatic audio sourced from this incredible [YouTube collection](https://ramanarunachalam.github.io/Music/Carnatic/carnatic.html).
### Key Features:
- Can identify **150 ragas**
- Does not require any information about the **shruthi (tonic pitch)** of the recording.
- **Compatible** with male/female vocal or instrumental recordings.

### Interpreting the Classifier:
We can gain an intuitive sense for what the classifier has learned. Here is a [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) projection of the hidden activations averaged per ragam. Each point is a ragam, and relative distances between the points indicate the degree to which the classifier thinks the ragas are similar. Each ragam is color coded by the [melakartha chakra](https://en.wikipedia.org/wiki/Melakarta#Chakras) it belongs to. We observe that the classifier has learned to a representation that roughly corresponds to these chakras! 



