import json
import utilities as utils
from langdetect import detect
from contractions import get_contractions
import argparse
import numpy as np
import pandas as pd
import nltk

parser = argparse.ArgumentParser(description="Specify number of reviews to parse")
parser.add_argument("-n", "--num_reviews", type=int, default=100000, help="Specify batch size for network")
parser.add_argument("-fb", "--file_bus", type=str, default='./data/dataset/business.json', help="Path to business json")
parser.add_argument("-fr", "--file_rev", type=str, default='./data/dataset/review.json', help="Path to review json")
parser.add_argument("-ds", "--download_stopwords", type=bool, default=True, help="Specify whether to download"
                                                                                 "NLTK stopwords")
parser.add_argument("-s", "--save_file", type=str, default="balanced_reviews.csv",
                    help="Specify file or path to save reviews to")
args = parser.parse_args()


def process_reviews(bus_file='.data/dataset/business.json', rev_file='./data/dataset/review.json'):
    """
    Function will initialize the review preprocessing pipeline. It will expand contractions of text
    and then perform text cleaning
    :param bus_file: Type string, path to business json file
    :param rev_file: Type string, path to reviews json file
    :return:
    """
    assert isinstance(bus_file, str)
    assert isinstance(rev_file, str)

    restId = []
    for line in open(bus_file, 'r'):
        data = json.loads(line)
        if 'Restaurants' in data['categories'] or 'Food' in data['categories']:
            restId.append(data['business_id'])
    print("There are %d restaurants" % (len(restId)))

    contractions = get_contractions()

    revs_list = [[]]
    stars_list = [[]]
    k = 0 # Count
    nolang = [[]]
    for line in open(rev_file, 'r'):    # encoding='utf-8'
        if k >= args.num_reviews:
            break
        data = json.loads(line)
        text = data['text']
        star = data['stars']
        ID = data['business_id']
        # Check language
        if text is None:
            continue
        if star is None:
            continue
        if ID not in restId:
            continue
        try:
            if detect(text) == 'en':
                revs_list.append(utils.clean_text(text, contractions))
                stars_list.append(star)
                k += 1
                # Notify for every 5000 reviews
                if len(revs_list) % 5000 == 0:
                    print(len(revs_list), k)
        except ValueError:
            nolang.append(text)
            print("Detected text with no language! Now at: %d" % len(nolang))
    print("Length of Reviews:\t" + str(len(revs_list)) + "Length of Stars:\t" +  str(len(stars_list)))
    return revs_list, stars_list


def drop_missing_data(df_reviews):
    """
    Helper function that will get rid of missing or incomplete or unusable data.
    This could include star ratings that aren't numeric, or without values or not within the specified range of 1-5
    :param df_reviews: Dataframe with categories 'stars' and 'text'
    :return: Dataframe with dropped values
    """
    print("Dropping Missing Data\n")
    df_reviews = df_reviews[np.isfinite(df_reviews['stars'])]
    df_reviews = df_reviews[np.isfinite(df_reviews['stars'])]
    df_reviews = df_reviews.dropna()
    df_reviews = df_reviews.reset_index(drop=True)
    return df_reviews


def convert_to_dataframe(revs_list, stars_list):
    """
    Helper function to convert the reviews and stars into a dataframe which can be easily managed and converted to csv
    :param revs_list: List of lists with each list containing the text review for restaurants
    :param stars_list: List of values with each value being a star rating for a review
    :return: Dataframe containing all reviews and stars in separate categorical columns
    """

    assert isinstance(revs_list, list)
    assert isinstance(stars_list, list)

    print("Converting to Dataframe")
    np_revs = np.asarray([revs_list]).T
    np_stars = np.asarray([stars_list]).T
    stacked_revs = np.hstack((np_revs, np_stars))
    categories = ['text', 'stars']
    df_reviews = pd.DataFrame(stacked_revs, columns=categories)
    df_reviews[['stars']] = df_reviews[['stars']].apply(pd.to_numeric)
    df_reviews = drop_missing_data(df_reviews)
    return df_reviews


def balance_dataframe(df, category=['stars']):
    """
    Function will balance the reviews so the network can have as even of training as possible
    Example) if there are 5000 total reviews, but the lowest amount of reviews is 500 one star reviews, the dataframe
    returned will have 500 of each review from 1 star to 5 stars

    :param df: pandas.DataFrame
    :param categorical_columns: iterable of categorical columns names contained in {df}
    :return: balanced pandas.DataFrame
    """
    if category is None or not all([col in df.columns for col in category]):
        raise ValueError('Please provide one or more columns containing categorical variables')

    lowest_count = df.groupby(category).apply(lambda x: x.shape[0]).min()
    df = df.groupby(category).apply(
        lambda x: x.sample(lowest_count)).drop(category, axis=1).reset_index().set_index('level_1')

    df.sort_index(inplace=True)

    return df


def main():
    if args.download_stopwords:
        nltk.download('stopwords')
    revs_list, stars_list = process_reviews(args.file_bus, args.file_rev)
    df_reviews = convert_to_dataframe(revs_list, stars_list)
    df_reviews['len'] = df_reviews.text.str.len()
    df_reviews = df_reviews[df_reviews['len'].between(10, 4000)]
    df_balanced = balance_dataframe(df_reviews)
    df_balanced.to_csv(args.save_file, encoding='utf-8')
    print("Done Processing %d reviews from %d" % (len(df_balanced['stars']), args.num_reviews))


if __name__ == "__main__":
    main()
