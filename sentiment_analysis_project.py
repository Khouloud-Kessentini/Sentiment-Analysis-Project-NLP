import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional
import nltk
import re
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('words')

# Initialize stop words and English words
stop_words = set(stopwords.words("english"))
english_words = set(words.words())

def preprocess_sentence(sentence):
    """Clean and preprocess a single sentence."""
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    sentence = re.sub(r'\d+', '', sentence)  # Remove numbers
    tokens = word_tokenize(sentence)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(filepath):
    """Load, clean, and preprocess the dataset."""
    # Load dataset
    df = pd.read_csv(filepath, header=None, names=['Class', 'Review'])
    df = df[:4000]
    
    # Preprocess the text
    df['ProcessedText'] = df['Review'].apply(preprocess_sentence)
    print("************************************************")
    print(df['Class'].value_counts())  # Show class distribution
    return df

def tokenize_text(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['ProcessedText'])
    sequences = tokenizer.texts_to_sequences(data['ProcessedText'])
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer, max_length

def encode_labels(data):
    labels = data['Class'] - 1  # Convert classes (1, 2) to (0, 1)
    return to_categorical(labels)

from keras.layers import Flatten

def build_model(input_dim, max_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=256, input_length=max_length),
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_and_predict(sentence, model, tokenizer, max_length):
    processed_sentence = preprocess_sentence(sentence)
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    return "Positive" if np.argmax(prediction) == 1 else "Negative"

def interactive_test(model, tokenizer, max_length):
    print("Enter a sentence to predict its sentiment ('exit' to quit):")
    while True:
        sentence = input("Sentence: ")
        if sentence.lower() == 'exit':
            break
        sentiment = preprocess_and_predict(sentence, model, tokenizer, max_length)
        print(f"Predicted Sentiment: {sentiment}")

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))

def main(train_filepath, test_filepath):
    train_data = load_and_preprocess_data(train_filepath)
    
    X_train, tokenizer, max_length = tokenize_text(train_data)
    y_train = encode_labels(train_data)
    
    test_data = load_and_preprocess_data(test_filepath)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['ProcessedText']), maxlen=max_length, padding='post')
    y_test = encode_labels(test_data)
    print(test_data)
    
    model = build_model(input_dim=len(tokenizer.word_index) + 1, max_length=max_length)
    model.fit(X_train, y_train, epochs=12, batch_size=16, validation_data=(X_test, y_test))
    
    evaluate_model(model, X_test, y_test)
    
    interactive_test(model, tokenizer, max_length)

if __name__ == "__main__":
    main("train.csv", "test.csv")
