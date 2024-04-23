user_tweet = input("Input your post: ")
# user_tweet = ["Everything that irritates us about others can lead us to an understanding of ourselves."] #payload['text']
from transformers import T5Tokenizer, T5ForConditionalGeneration

def extract_keywords(tweets):
    # Load the model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords")
    tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords")
    # model.cuda()
    # In case we need name entity extraction instead of keywords extraction
    # from transformers import pipeline
    # pipe = pipeline("token-classification", model="dslim/bert-large-NER")


    task_prefix = "Keywords: "
    results = []

    for tweet in tweets:
        # Prepare the input for the model
        input_sequences = [task_prefix + tweet]
        input_ids = tokenizer(input_sequences, return_tensors="pt", truncation=True).input_ids

        # Generate output from the model
        output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)

        # Decode the output to get the keywords
        predicted_keywords = tokenizer.decode(output[0], skip_special_tokens=True)

        # Store the result
        results.append(predicted_keywords)

    return results

keywords = extract_keywords(user_tweet)
modified_strings = [s.replace(',', ' OR') for s in keywords]


import requests
import os
import json
import re
from sentence_transformers import SentenceTransformer

# Assuming you have set your environment variable as BEARER_TOKEN
bearer_token = ""

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# model.cuda()
search_url = "https://api.twitter.com/2/tweets/search/all"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {
    'query': '(from:twitterdev -is:retweet) OR '+ modified_strings[0],
    'tweet.fields': 'author_id',
    'max_results': '500'  # Adjust the number of results per call, max is usually 100 for standard API
}

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    json_response = connect_to_endpoint(search_url, query_params)
    json.dumps(json_response, indent=4, sort_keys=True)

    tweet_texts = [tweet['text'] for tweet in json_response['data']] if 'data' in json_response else []

    # Output the list of tweet texts
    # print(tweet_texts)

    embeddings = model.encode(tweet_texts)
    embeddings1 = model.encode(user_tweet)


    return embeddings, embeddings1, tweet_texts


vectors, vectors1, X_tweets = main()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gaussian_kde

# Set the aesthetic style of the plots
sns.set(style="whitegrid", context="talk", palette="muted")
plt.style.use('bmh')

# Generate random vectors (for example, 100 vectors in 5-dimensional space)


# Calculate pairwise Euclidean distances among existing vectors
distances = pdist(vectors, 'euclidean')

# Assume a new vector is given
new_vector = vectors1.reshape(1,-1)

# Calculate distances from this new vector to all existing vectors
new_vector_distances = cdist(new_vector, vectors)[0]
mean_new_vector_distance = np.mean(new_vector_distances)  # Calculate the mean distance

# Plotting the distribution of existing vector distances
plt.figure(figsize=(10, 6), dpi=100)
ax = sns.histplot(distances, kde=True, color='skyblue', label='Samples of Perspective', element='step', stat='density')

# Find the y-value on the KDE curve at the mean distance
kde = gaussian_kde(distances)
density_at_mean = kde(mean_new_vector_distance)

# Add an arrow pointing to the mean distance on the KDE curve
plt.annotate(
    '', xy=(mean_new_vector_distance, density_at_mean), xytext=(mean_new_vector_distance, density_at_mean + 0.01),
    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8)
)
plt.text(mean_new_vector_distance + 0.1, density_at_mean + 0.01, 'Your post', verticalalignment='center', color='red')

plt.title(f'X Spectrum: {modified_strings}')
plt.xlabel('Spectrum')
plt.ylabel('Volume')
plt.legend()
plt.show()
import json



# Save to a JSON file
data = {'platform_tweets': X_tweets, 'distances': distances.tolist(), 'user_tweet': user_tweet, 'user_tweet_distance':mean_new_vector_distance}
with open('data.json', 'w') as f:
    json.dump(data, f)

