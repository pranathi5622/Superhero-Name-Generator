
## Project Title: Superhero Name Generator

### Objective

The objective of this project is to develop a deep learning model capable of generating novel superhero and supervillain names. The model is trained on an existing dataset of character names and learns to predict sequences of characters that form names similar in style and structure to those in the dataset.

---

## Dataset Description

The dataset consists of a text file containing thousands of superhero and supervillain names. Each name is separated by a tab character (`\t`) which is used to indicate the end of a name. The names vary in length and format, and the dataset is used to train the model to learn patterns in name formation.

---

## Preprocessing Steps

### Tokenization

Each name is treated as a sequence of characters. A tokenizer is configured to split the text on characters (not words), and special characters such as punctuation are excluded. The tokenizer constructs a vocabulary of all unique characters present in the dataset.

Two important mappings are created:

* `char_to_index`: Maps each character to a unique integer.
* `index_to_char`: Maps each integer back to its corresponding character.

### Sequence Conversion

Each name is converted into a sequence of integers based on the tokenizer's character-to-index mapping. These sequences are then used to generate multiple sub-sequences. For a name of length *n*, we create *n - 1* training samples by slicing increasing prefixes of the sequence. This allows the model to learn the probability distribution of the next character given the previous characters.

For example, for the name represented by `[25, 16, 12, 20, 2, 1]`, the following training sequences are created:

```
[25, 16]
[25, 16, 12]
[25, 16, 12, 20]
...
```

### Padding

To ensure all sequences have the same length, we apply pre-padding using zeros. The maximum sequence length observed in the dataset is used as the standard length for all sequences.

---

## Data Preparation

The dataset is then split into input (`x`) and target (`y`) values. Each input sequence consists of all but the last character, while the target value is the next character in the sequence. The final dataset is split into training and validation subsets using an 80:20 ratio.

---

## Model Architecture

The model is built using the Keras Sequential API. It includes the following layers:

1. **Embedding Layer**: Converts each integer (representing a character) into a dense vector of fixed size (8 dimensions). This allows the model to learn a distributed representation of characters.
2. **Convolutional Layer (Conv1D)**: Applies causal 1D convolution to capture local patterns in character sequences, using 64 filters and a kernel size of 5. The causal padding ensures the model does not violate the temporal order.
3. **MaxPooling Layer**: Reduces the spatial size of the output from the convolutional layer by taking the maximum value over a window, thereby decreasing computation and controlling overfitting.
4. **LSTM Layer**: A Long Short-Term Memory layer with 32 units is used to capture long-range dependencies in the sequence.
5. **Dense Layer**: Outputs a probability distribution over all characters using the softmax activation function.

The model is compiled using the Adam optimizer and the sparse categorical crossentropy loss function. Accuracy is used as the evaluation metric.

---

## Training Process

The model is trained on approximately 88,000 character sequences for a maximum of 50 epochs. Early stopping is implemented to prevent overfitting. Training is halted if the validation accuracy does not improve for three consecutive epochs.

---

## Name Generation

To generate a new name, a seed string is provided (e.g., `"kur"`). This seed is converted into a padded sequence of integers. The model then predicts the most likely next character. This predicted character is appended to the seed, and the process is repeated until either the end-of-name character (`\t`) is predicted or a maximum length (e.g., 40 characters) is reached.

This method allows the generation of creative and coherent superhero names that were not present in the training dataset.

---

## Conclusion

This project demonstrates how sequence modeling techniques in deep learning can be applied to the creative task of name generation. The model successfully learns structural patterns in superhero and supervillain names and can generate new names with similar linguistic characteristics. The architecture combines embedding, convolutional, and recurrent layers to achieve effective character-level language modeling.

---


