# Sentiment-Analysis-Project-NLP

This project implements a **sentiment analysis system** that classifies text reviews as either **positive** or **negative** using deep learning techniques. The system involves several stages, including data preprocessing, tokenization, model training, and prediction. The core of the project is built using TensorFlow/Keras, with the model employing an embedding layer, dense layers, and dropout layers for effective learning and regularization.

The project covers the following main areas:
 * **Preprocessing**: Cleaning and tokenizing text data, removing stopwords, punctuation, and numbers.
 * **Tokenization and Encoding**: Transforming text into numerical sequences and ensuring uniform input size.
 * **Model Architecture**: Constructing a neural network for classification with an embedding layer for text representation.
 * **Optimization**: Using the Adam optimizer to minimize the categorical cross-entropy loss function.
 * **Interactive Testing**: Enabling user input for real-time sentiment prediction after training.

The goal of this sentiment analysis task is to learn a function $$f: X \to Y$$, where:

$$
X \subset \mathbb{R}^d \quad \text{and} \quad Y = \{0, 1\}
$$

Here, $$X$$ represents the feature space of text reviews, where each review is a sequence of words transformed into a numerical vector. The set $$Y$$ contains the sentiment labels, where $$Y = 0$$  if the review is negative and  $$Y = 1$$  if the review is positive.

The function $$f$$ is modeled using a deep learning approach, specifically a neural network, which is trained to map from the input text features $$X$$ to the sentiment labels $$Y$$.

# Model Architecture

The model architecture is a **feedforward neural network** with the following components:
 1. Hidden Layers
   * ReLu activation
    $$h_1 = \text{ReLU}(W_1x + b_1)$$
   * Dropout is applied to prevent overfitting.
 2. Output Layer (Softmax)
    $$\hat{y} = \text{softmax}(W_2h_1 + b_2)$$

# Metrics

The accuracy is computed as:

$$\text{Accuracy}  = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

# Training and Validation

The model trains over 15 epochs, minimizing the loss function and updating weights using Adam's gradient-based optimization.


### How to Run

Clone the repository :

```python
git clone https://github.com/Khouloud-Kessentini/Sentiment-Analysis-Project-NLP.git
cd Sentiment-Analysis-Project-NLP
```

Run the script
```python
sentiment_analysis_project.py
```

### Contributing

Contributions are welcome ! Feel free to open issues or submit pull requests to improve this project.
