
## Project Title: Superhero Name Generator
This project generates superhero and supervillain names using a character-level language model trained on a dataset of existing names.

## Workflow

### Data Loading and Preprocessing

* Load text data and split by newlines to extract individual names.
* Use `Tokenizer` to tokenize characters, filtering special characters.
* Create `char_to_index` and `index_to_char` dictionaries.
* Convert names to sequences of character indices.
* Generate input-output pairs where inputs are prefix subsequences and outputs are the next character.
* Pad sequences to the maximum length.

### Model

* Input: padded sequences of character indices.
* Layers:

  * `Embedding(num_chars, 8)`
  * `Conv1D(64, kernel_size=5, activation='tanh', padding='causal')`
  * `MaxPooling1D(pool_size=2)`
  * `LSTM(32)`
  * `Dense(num_chars, activation='softmax')`
* Compile with:

  * Loss: `sparse_categorical_crossentropy`
  * Optimizer: `adam`
  * Metrics: `accuracy`

### Training

* Split data into training and validation sets.
* Train for up to 50 epochs with early stopping based on validation accuracy.

### Name Generation

* Start with a seed string.
* Predict and append one character at a time using the trained model.
* Stop after 40 iterations or upon predicting the tab character.

## Requirements

* Python
* TensorFlow
* scikit-learn

---

Let me know if you want to include a sample output or usage instructions.


