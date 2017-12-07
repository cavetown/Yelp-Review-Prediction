# Yelp Review Data Analysis

## Project Overview
On Yelp, a restaurant can have over hundreds of reviews, and it is difficult for users to read through them all. As a result, the star rating is usually the most important criterion for users when they make judgments. However, the relationship between text reviews and star rating is complicated and unobvious. To tackle this popular NLP problem, I've decided to use a recurrent neural network to predict star ratings from reviews.

## Approach
1. **Dataset**: We will be using the dataset for the Yelp dataset challenge (https://www.yelp.com/dataset/challenge). The dataset is in JSON format. The important components for each review JSON string would be what’s stored in “text”, “stars”.

2. **Preprocessing**: To parse the JSON file into Python, I remove common stopworks (from NLTK's english stopwords list) as well as other common preprocessing procedures (lower case everything, remove punctuation, leave english only reviews, etc). Because restaurants make up a majority of the reviews found on Yelp, I decide to filter out only reviews for business's that were considered restaurants. I also balance the reviews so there is an equal representation of all star ratings in both training and test sets.

3. **Prediction**: For the module training, we will use pretrained ConceptNet Numberbatch embeddings (https://github.com/commonsense/conceptnet-numberbatch) and an 2-Layer GRU RNN in order to predict the star rating for a review. The main framework for this project is TensorFlow. 

## Usage
The main file to run is tf_rnn_yelp.ipynb for training. It is a jupyter notebook file so it will require jupyter installed. 
The file is self sustained and the only things required to run are the datasets and embedding (which there is a link for in the markdowns). Markdowns inside the notebook explain each procedure and what it is accomplishing along with variables easily modifiable.
