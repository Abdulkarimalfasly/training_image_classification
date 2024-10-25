# Importing the required libraries
import pandas as pd  # Pandas library for comfortable data management and analysis
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.feature_extraction.text import CountVectorizer  # To convert text into numerical representation (matrix of numbers)
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes model used for classification
from sklearn.metrics import accuracy_score  # To calculate the model's accuracy
import nltk  # NLTK library specialized in natural language processing
from nltk.corpus import movie_reviews  # Importing the movie reviews dataset from NLTK

# Downloading the movie reviews dataset from the NLTK library
nltk.download('movie_reviews')  # Download the dataset if it is not already downloaded

# Preparing the data
documents = [(list(movie_reviews.words(fileid)), category)  
             for category in movie_reviews.categories()  
             for fileid in movie_reviews.fileids(category)]  

# Converting data into a DataFrame
data = pd.DataFrame(documents, columns=['Review', 'Sentiment'])

# Displaying the first few rows for a preview of the DataFrame
print(data.head())  

# Converting texts to string format
data['Review'] = data['Review'].apply(lambda x: ' '.join(x))  

# Displaying the data after conversion
print(data.head())  

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Sentiment'], 
                                                    test_size=0.2, random_state=42)

# Displaying the sizes of the splits
print(f'Train size: {len(X_train)}')  
print(f'Test size: {len(X_test)}')    

# Converting texts to numbers using CountVectorizer
vectorizer = CountVectorizer()  
X_train_vec = vectorizer.fit_transform(X_train)  
X_test_vec = vectorizer.transform(X_test)  

# Displaying the shape of the transformed matrices
print(f'X_train shape: {X_train_vec.shape}')  
print(f'X_test shape: {X_test_vec.shape}')    

# Building the Naive Bayes model
model = MultinomialNB()  
model.fit(X_train_vec, y_train)  

# Predicting sentiments using the model
y_pred = model.predict(X_test_vec)  

# Calculating the model's accuracy
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy * 100:.2f}%')  

# Testing the model on a new text
new_review = ["I absolutely loved this movie, it was fantastic and full of great moments!"]
new_review_vec = vectorizer.transform(new_review)  
prediction = model.predict(new_review_vec)  
print(f"Sentiment: {prediction[0]}")  

# Analyzing more new texts
new_reviews = [
    "This movie was terrible and a waste of time.",
    "What a great film! I really enjoyed it.",
    "I would not recommend this movie to anyone.",
    "An amazing experience, truly a masterpiece!",
    "The plot was boring and the characters were flat."
]

# Predicting each new review and displaying the results
for review in new_reviews:
    review_vec = vectorizer.transform([review])  
    pred = model.predict(review_vec)  
    print(f"Review: '{review}' -> Sentiment: {pred[0]}")
