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
### Python Files
1. Ensure that all requirements and dependencies are installed
``` 
pip install -r requirements.txt
```
2. Run preprocessing file to process reviews
```
python preprocess_data.py -n=int -fb=str -fr=str -ds=bool -s=str
```
-n, --num_reviews: amount of reviews to process  
-fb, --file_bus: path to business json file from Yelp dataset  
-fr, --file_rev: path to review jason file from Yelp dataset  
-ds, --download_stopwords: download NLTK stopwords if they are't downloaded already  
-s, --save_file: where to save the preprocessed data as a csv file  

3. Run predict_ratings.py
```
python predict_ratings.py -nl=int -bz=int -e=int -hu=int -t=str -kp=float -ep=str -f=str -ed=int -v=float -p=str -r=str -lrd=float -lr=float -s=str -uc=int
```
-nl, --num_layers: specify layers for network  
-bz, --batch_size: how much data to feed in per time  
-e, --epochs: total epochs to train for  
-hu, --hidden_units: hidden size of GRU  
-t, --task: 'train', 'test', or 'predict'  
-kp, --keep_prob: how much to keep during dropout  
-ep, --embedding_path: path to embeddings used  
-f, --file: path to CSV from preprocess  
-ed, --embedding_dim: dimension of embedding matrix  
-v, --val_split: how much data to split into test (validation) set  
-p, --pickle: specify whether to pickle files used  
-r, --resume: resume a pretrained model  
-lrd, --learning_rate_decay: how much to anneal learning rate  
-lr, --learning_rate: learning rate to use  
-s, --shuffle: whether to shuffle data after every epoch  
-uc, --update_check: how often to check current loss and to save model

### Results:  
Final results for the network was able to achieve 62% accuracy with the spread of:
![alt text](https://github.com/thomasan95/Yelp-Review-Prediction/blob/master/figures/cmatrix.png?raw=true)
