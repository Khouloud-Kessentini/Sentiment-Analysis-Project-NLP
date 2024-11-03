# Sentiment-Analysis-Project-NLP

The main steps include:

1.   Data pre-processing

  *   Data loading and columns selection
  *   Convert reviews to lowercase and remove emojis, numbers and punctuation
  *   Remove stop words ("and", "in", "at", etc)

2.   One-Hot encoding of reviews (conversion of text reviews to boolean vectors)
3.   Splitting data into test and evaluation (cross-validation 80% - 20%)
4.   Neural network implementation
  * Architecture:
      *  Input layer size  = number of variables (columns) in the One-Hot matrix
      *  Hidden layer 1 size = 64
      *  Hidden layer 2 size = 128
      *  Output layer size = 2 (number of classes, or potential reviews; either 0 or 1)
5.   Adding dropout to prevent overfitting
6.   Testing model accuracy

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
# Download necessary NLTK data
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('words')
# Initialize stop words and English words globally
stop_words = set(stopwords.words("english"))
english_words = set(words.words())
```

## Data preprocessing
### Data loading
```python
# loading data
df = pd.read_csv("/content/full-corpus.csv")
df
```
<img src="img/1 - initial_data.png" alt="Project Logo" width="2000" height="444"/>

### Sentiment column digitalization
```python
def load_and_preprocess_data(filepath):
    """Load data from a CSV file, preprocess it, and prepare for training."""
    df = pd.read_csv(filepath)
    data = df.loc[df['Sentiment'].isin(['positive', 'negative'])].copy()
    data.reset_index(drop=True, inplace=True)
    data['Sentiment'] = data['Sentiment'].replace({'positive': 0, 'negative': 1})
    data['ProcessedText'] = data['TweetText'].apply(preprocess_sentence)
    return data
```

### Tweets before preprocessing
<img src="img/2 - tweets_before_preprocessing.png" alt="Tweets before preprocessing" width="2000" height="200"/>

### Tweets preprocessing
```python
# loading english stopwords and words
stop_words = set(stopwords.words("english"))
english_words = set(words.words())

def preprocess_sentence(sentence):
    """Preprocess the input sentence by cleaning, tokenizing, and filtering words."""
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation/emojis
    sentence = re.sub(r'\d+', '', sentence)  # Remove numbers
    sentence = re.sub(r'(.)\1{2,}', r'\1', sentence)  # Replace repeated characters
    tokens = word_tokenize(sentence)
    tokens = [word for word in tokens if word in english_words and len(word) > 2]  # Keep valid words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)
```

### Tweets after preprocessing
<img src="img/3 - tweets_after_preprocessing.png" alt="Tweets after preprocessing" width="2000" height="200"/>

## OneHot Data Encoding
```python
def vectorize_text(data):
    """Vectorize the processed text data using one-hot encoding."""
    vectorizer = CountVectorizer()
    one_hot_matrix = vectorizer.fit_transform(data['ProcessedText'])
    one_hot_matrix_df = pd.DataFrame(one_hot_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return one_hot_matrix_df, vectorizer

def encode_labels(data):
    """Encode sentiment labels to categorical format for training."""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['Sentiment'])
    categorical_labels = to_categorical(encoded_labels, num_classes=2)
    return categorical_labels
```
<img src="img/4 - tweets_one_hot_encoding.png" alt="Tweets after preprocessing" width="2000" height="400"/>

## Encode the sentiment column
```python
def encode_labels(data):
    """Encode sentiment labels to categorical format for training."""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['Sentiment'])
    categorical_labels = to_categorical(encoded_labels, num_classes=2)
    return categorical_labels
```

## Construct the neural network (Multi-layer perceptron)
```python
def build_model(input_shape):
    """Build and compile a neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
## Train the model and evaluate it
```python
def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with the provided training and validation data."""
    history = model.fit(X_train, y_train, epochs=30, batch_size=2, validation_data=(X_test, y_test))
    return history
```
<img src="img/5 - results.png" alt="results.png" width="2000" height="599"/>

## Evaluate the model
```python

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy
```

## Test the model (make "positive" or "negative" sentiment prediction)
```python
def preprocess_and_predict(sentence, model, vectorizer):
    """Preprocess the sentence and predict sentiment using the trained model."""
    processed_sentence = preprocess_sentence(sentence)
    sentence_vector = vectorizer.transform([processed_sentence]).toarray()
    prediction = model.predict(sentence_vector)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "Positive" if predicted_class == 0 else "Negative"

def interactive_test(model, vectorizer):
    """Continuously prompt for input and predict sentiment until user stops."""
    print("Enter a sentence to predict its sentiment ('exit' to quit):")
    while True:
        test_sentence = input("Sentence: ")
        if test_sentence.lower() == 'exit':
            print("Exiting.")
            break
        predicted_sentiment = preprocess_and_predict(test_sentence, model, vectorizer)
        print(f"The sentiment for the input sentence is: {predicted_sentiment}")

def main(filepath):
    """Main function to execute the training and prediction process."""
    download_nltk_data()
    data = load_and_preprocess_data(filepath)
    one_hot_matrix_df, vectorizer = vectorize_text(data)
    categorical_labels = encode_labels(data)
    X_train, X_test, y_train, y_test = train_test_split(one_hot_matrix_df, categorical_labels, test_size=0.3, random_state=42)
    model = build_model(input_shape=X_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    interactive_test(model, vectorizer)

# Execute the main function with the file path to the dataset
if __name__ == "__main__":
    main("/content/full-corpus.csv")
```
