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
sentiment = df[['Sentiment']].copy()  # sentiment column selection
sentiment = sentiment[:480]  # selecting a subset of the data rows
# digitalizing sentiment column
for i in range(sentiment.shape[0]):
  if sentiment['Sentiment'][i] == "positive":
    sentiment['Sentiment'][i] = 0
  if sentiment['Sentiment'][i] == "negative":
    sentiment['Sentiment'][i] = 1
# verifying numerization
print(sentiment['Sentiment'].unique())
```
### Tweets data selection
```python
reviews = df[['TweetText']].copy()
reviews = reviews[:480]
reviews.columns = ['reviews']
```

### Tweets before preprocessing
<img src="img/2 - tweets_before_preprocessing.png" alt="Tweets before preprocessing" width="2000" height="200"/>

### Tweets preprocessing
```python
# loading english stopwords and words
stop_words = set(stopwords.words("english"))
english_words = set(words.words())

# tweets preprocessing
for i in range(reviews.shape[0]):
  reviews['reviews'][i] = reviews['reviews'][i].lower() # convert to lowercase
  reviews['reviews'][i] = re.sub(r'[^\w\s]', '', reviews['reviews'][i]) # remove punctuation and emojis
  reviews['reviews'][i] = re.sub(r'\d+', '', reviews['reviews'][i]) # remove numbers
  reviews['reviews'][i] = re.sub(r'[^a-zA-Z\s]', '', reviews['reviews'][i])
  reviews['reviews'][i] = re.sub(r'(.)\1{2,}', r'\1', reviews['reviews'][i])
  tokens = word_tokenize(reviews['reviews'][i])
  tokens = [word for word in tokens if word in english_words and len(word) > 2] # remove non english words
  filtered_tokens = [word for word in tokens if word not in stop_words] # remove stop words
  reviews['reviews'][i] = ' '.join(filtered_tokens)
```

### Tweets after preprocessing
<img src="img/3 - tweets_after_preprocessing.png" alt="Tweets after preprocessing" width="2000" height="200"/>

## OneHot Data Encoding
```python
# initialize vectorizer
vectorizer = CountVectorizer()
# create one-hot matrix
one_hot_matrix = vectorizer.fit_transform(reviews['reviews'])
one_hot_matrix_df = pd.DataFrame(one_hot_matrix.toarray(), columns=vectorizer.get_feature_names_out())
one_hot_matrix_df = one_hot_matrix_df
```
<img src="img/4 - tweets_one_hot_encoding.png" alt="Tweets after preprocessing" width="2000" height="400"/>

## Encode the sentiment column
```python
# Encode the labels (sentiment)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sentiment['Sentiment'])
categorical_labels = to_categorical(encoded_labels, num_classes=2)
```
## Split the data into train and test data (cross-validation 70-30)
```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(one_hot_matrix_df, categorical_labels, test_size=0.3, random_state=42)
```

## Construct the neural network (Multi-layer perceptron)
```python
# Build the neural network model
model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.1))  # Dropout to reduce overfitting

# Hidden Layer 2
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))  # Dropout to reduce overfitting

# Output Layer
model.add(Dense(2, activation='softmax'))  # 2 output classes for sentiment

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```
## Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test))
