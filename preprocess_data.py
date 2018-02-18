import json
import utilities as utils
from langdetect import detect
from contractions import get_contractions
import argparse

parser = argparse.ArgumentParser(description="Specify number of reviews to parse")
parser.add_argument("-n", "--num_reviews", type=int, default=100000, help="Specify batch size for network")
args = parser.parse_args()

def process_reviews(bus_file='.data/dataset/business.json', rev_file='./data/dataset/review.json'):
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
        if text == None:
            continue
        if star == None:
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
        except:
            nolang.append(text)

    return revs_list, stars_list
