# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Import packages
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the necessary NLTK packages - first time use only
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Open and read in the corpus of words
f = open("C:/Users/kobys/Desktop/AllSeus.txt", "r", errors = "ignore")
raw = f.read()

# SET GLOBAL VARIABLES
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
similarityCutOff = 0.3

# Pre-process the data
raw = raw.lower() # convert corpus to lowercase
sent_tokens = nltk.sent_tokenize(raw) # convert corpus to a list of sentences 
word_tokens = nltk.word_tokenize(raw) # convert corpus to a list of words

lemmer = nltk.stem.WordNetLemmatizer() # create lemmatizer object
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) # remove punctuation

# Function to lemmatize the words in the corpus
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Function to LemNormalize the words in the corpus
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Function to select a greeting response
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Function for chatbot response
def response(user_response):
    # Chatbot setup
    robo_response = '' # Create a blank chatbot respons string
    sent_tokens.append(user_response) # Append the user's response to the end of the sentence tokens list
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') # Assign weights to the words based on their frequency/importance
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Find the best response for the chatbot
    vals = cosine_similarity(tfidf[-1], tfidf)
    # print("cosine similarity vals")
    # print(vals)

    # Grab the index of the top reponse - ignoring the user input
    idx = vals.argsort()[0][-2]

    # # Play around with other close responses
    # idx1 = vals.argsort()[0][-3]
    # idx2 = vals.argsort()[0][-4]

    # # Create a list of the top three responses & randomly choose one
    # idList = [idx-1,idx1-1,idx2-1]    
    # idT = random.choice(idList) 

    # Get all the values in a 1-D list & sort the scores
    flat = vals.flatten()
    flat.sort() # Sort scores in ascending order
    sim_score = flat[-2] # Grab the best similarity score - ignoring the user input

    # # Check the similiarity score
    # print("Similarity Score")
    # print(sim_score)

    # Check if a response matched at above the similiarity cutoff score set above
    if(sim_score < similarityCutOff):
        robo_response = "Thing does not understand. Please rephrase."
        return robo_response
    else:
        robo_response = sent_tokens[idx] # Return a response
        return robo_response

# Create a function to talk with the chatbot
def talk():
    # Start the conversation
    print("ChatBot: My name is Thing. Have a chat with me! If you want to exit, type 'Bye'")
    while(True):
        
        print()
        # Replace "Name" with your name
        print("Name: ", end="")

        # Grab the user input and convert it to lowercase
        user_response = input()
        user_response = user_response.lower()

        # Carry on the conversation
        if(user_response != 'bye'):
            if(user_response =='thanks' or user_response=='thank you' or user_response=='gracias' ):
                print("ChatBot: You are welcome!!")
                break
            else:
                if(greeting(user_response) != None):
                    print("ChatBot: " +greeting(user_response))
                else:
                    print("ChatBot: ", end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response) # Remove user input before continuing the conversation
        else:
            print("ChatBot: Ciao!")
            break

# Run the program
def main():

    talk()

main()