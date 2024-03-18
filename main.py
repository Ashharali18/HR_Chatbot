import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
import csv

warnings.filterwarnings('ignore')
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Exploratory data analysis
df = pd.read_csv('UpdatedResumeDataSet.csv')
category = df['Category'].value_counts().reset_index()


# Cleaning the data
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                        resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


df['cleaned'] = df['Resume'].apply(lambda x: cleanResume(x))

# getting the entire resume text
corpus = " "
for i in range(0, len(df)):
    corpus = corpus + df["cleaned"][i]

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
import string
from wordcloud import WordCloud

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
# Tokenizing the text
tokens = tokenizer.tokenize(corpus)

# now we shall make everything lowercase for uniformity
# to hold the new lower case words
words = []
# Looping through the tokens and make them lower case
for word in tokens:
    words.append(word.lower())

# Now encode the data
label = LabelEncoder()
df['new_Category'] = label.fit_transform(df['Category'])

# Vectorizing the cleaned columns
text = df['cleaned'].values
target = df['new_Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(text)
WordFeatures = word_vectorizer.transform(text)

# Separate train and test data
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, target, random_state=24, test_size=0.2)

# Model Training
model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X_train, y_train)


def TopResumes():
    # Ask user for input category
    input_category = input("Enter a category: ")
    if input_category not in df['Category'].values:
        print("Error: Category not found in the dataset.")
    else:
        # Predict category of resumes belonging to input category
        mask = df['Category'] == input_category
        input_resumes = df[mask]['cleaned'].values
        input_targets = df[mask]['new_Category'].values
        input_word_features = word_vectorizer.transform(input_resumes)

        # Predictions and probabilities
        predictions = model.predict(input_word_features)
        probabilities = model.predict_proba(input_word_features)

        # Create a list of tuples with probabilities and cleaned resumes
        resume_probs = []
        for i in range(len(input_resumes)):
            predicted_category = label.inverse_transform([predictions[i]])[0]
            probability = probabilities[i][predictions[i]]
            resume_probs.append((probability, input_resumes[i], predicted_category))

        # Sort the list of tuples based on probability in descending order
        resume_probs = sorted(resume_probs, key=lambda x: x[0], reverse=True)

        # Print the top 10 sorted resumes with predicted categories and probabilities
        print("\nTop 10 Resumes:")
        for i, (probability, resume_text, predicted_category) in enumerate(resume_probs[:10]):
            index = df[mask].index[i] + 2
            print(f"\nResume {index}: \n{resume_text}")
            print(f"Predicted Category: {predicted_category}")
            print(f"Probability of Prediction: {probability:.2f}")


def MakeRecommendation():
    user_resume = input("Please enter your resume: ")
    desired_category = input("Please enter the category you're interested in: ")

    # Clean the user input and vectorize it
    cleaned_resume = cleanResume(user_resume)
    user_resume_vector = word_vectorizer.transform([cleaned_resume])

    # Make prediction
    predicted_category = label.inverse_transform(model.predict(user_resume_vector))[0]

    if predicted_category == desired_category:
        print("The applicant is recommended for this job")
    else:
        print("The applicant is not recommended for this job")


# read in the CSV file
with open('Conversational_Data.csv', mode='r') as file:
    reader = csv.reader(file)
    qa_pairs = {rows[0]: rows[1] for rows in reader}


# preprocess the text data
def preprocess(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove digits
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()


# create a vectorizer to convert the text data into vectors
vectorizer = TfidfVectorizer(preprocessor=preprocess)

# fit the vectorizer on the question-answer pairs in the dataset
corpus = list(qa_pairs.keys()) + list(qa_pairs.values())
vectorizer.fit(corpus)


# define a function to find the most similar question-answer pair
def get_most_similar_question(user_input):
    # preprocess the user's input
    preprocessed_input = preprocess(user_input)

    # convert the user's input into a vector
    input_vector = vectorizer.transform([preprocessed_input])

    # compute the cosine similarity between the user's input and each question in the dataset
    similarity_scores = []
    for question in qa_pairs.keys():
        question_vector = vectorizer.transform([question])
        similarity_score = cosine_similarity(input_vector, question_vector)[0][0]
        similarity_scores.append(similarity_score)

    # find the index of the question with the highest similarity score
    max_index = similarity_scores.index(max(similarity_scores))

    # return the corresponding answer
    return qa_pairs[list(qa_pairs.keys())[max_index]]

def FAQ():
    print("Hello! How may I assist you?")
    while True:

        user_input = input("User: ")
        if user_input == "exit":
            break
        else:
            answer = get_most_similar_question(user_input)
            print("Chatbot:", answer)

user_choice = input("Enter 1 for HR, Enter 2 for Employee: ")
if user_choice == "1":
    user_choice1 = input("Enter 1 get top Resume, Enter 2 for makind Recommendation: ")
    if user_choice1 == "1":
        TopResumes()
    elif user_choice1 == "2":
        MakeRecommendation()
elif user_choice == "2":
    FAQ()



