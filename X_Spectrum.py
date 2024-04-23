import json
import requests
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gaussian_kde

# Constants for API authentication
BEARER_TOKEN = ""  # Place your Twitter Bearer Token here

def extract_keywords_from_tweets(tweets):
    """
    This function extracts keywords from a list of tweets using a pre-trained T5 model.
    
    Args:
        tweets (list of str): The list of tweets from which to extract keywords.
        
    Returns:
        list of str: A list of extracted keywords for each tweet.
    """
    
    # Initialize the model and tokenizer with a pre-trained keywords extraction model
    keyword_model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords")
    keyword_tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords")
    
    # List to store results of keyword extraction
    keyword_results = []
    
    # Prefix to guide the model's predictions
    task_prefix = "Keywords: "  

    # Process each tweet to extract keywords
    for tweet in tweets:
        # Prepare the input for the model
        input_sequence = [task_prefix + tweet]
        
        # Tokenize the tweet text
        input_ids = keyword_tokenizer(input_sequence, return_tensors="pt", truncation=True).input_ids
        
        # Generate keywords using the model
        keyword_output = keyword_model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
        
        # Decode the generated keywords
        predicted_keywords = keyword_tokenizer.decode(keyword_output[0], skip_special_tokens=True)
        
        # Append the predicted keywords to the results list
        keyword_results.append(predicted_keywords)

    return keyword_results

def authenticate_request(request):
    """
    This method adds bearer token authentication to requests sent to the Twitter API.
    
    Args:
        request (requests.PreparedRequest): The outgoing request object to modify.
    
    Returns:
        requests.PreparedRequest: The modified request object with authentication details added.
    """
    
    # Add bearer token to the request headers
    request.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    
    # Set User-Agent for the request
    request.headers["User-Agent"] = "v2FullArchiveSearchPython"
    
    return request

def fetch_tweets(url, query_params):
    """
    This function connects to the Twitter API using the provided URL and parameters to fetch tweets.
    
    Args:
        url (str): The API endpoint URL.
        query_params (dict): The query parameters for the API request.
    
    Returns:
        dict: The JSON response from the API.
    
    Raises:
        Exception: If the API response status is not 200.
    """
    
    # Perform the GET request to the Twitter API
    response = requests.get(url, auth=authenticate_request, params=query_params)
    
    # Check for successful response
    if response.status_code != 200:
        raise Exception(f"HTTP Status Code {response.status_code}: {response.text}")
    
    return response.json()

def main():
    """
    Main function to execute the workflow: keyword extraction, API connection, and embedding calculation.
    """
    
    # Prompt user for input
    user_input_tweet = input("Input your post: ")
    
    # Extract keywords from the user's tweet
    extracted_keywords = extract_keywords_from_tweets([user_input_tweet])
    
    # Modify keywords for search query
    modified_keywords = [keywords.replace(',', ' OR') for keywords in extracted_keywords]

    # Initialize the embedding model
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    
    # Set API endpoint and parameters
    api_search_url = "https://api.twitter.com/2/tweets/search/all"
    api_query_params = {
        'query': '(from:twitterdev -is:retweet) OR ' + modified_keywords[0],
        'tweet.fields': 'author_id',
        'max_results': '500'
    }

    # Fetch tweets using the API
    api_response = fetch_tweets(api_search_url, api_query_params)
    
    # Extract text from tweets
    tweet_texts = [tweet['text'] for tweet in api_response['data']] if 'data' in api_response else []
    
    # Calculate embeddings for tweets
    tweet_embeddings = embedding_model.encode(tweet_texts)
    
    # Calculate embedding for user's tweet
    user_tweet_embedding = embedding_model.encode([user_input_tweet])

    return tweet_embeddings, user_tweet_embedding, tweet_texts

def plot_distance_distribution(tweet_vectors, user_tweet_vector):
    """
    This function plots the distribution of Euclidean distances between tweet embeddings.
    
    Args:
        tweet_vectors (np.array): Array of tweet embeddings.
        user_tweet_vector (np.array): Array of the user's tweet embedding.
    """
    
    # Calculate pairwise distances between all tweet vectors
    distances = pdist(tweet_vectors, 'euclidean')
    
    # Calculate distances between user's tweet and all other tweets
    user_vector = user_tweet_vector.reshape(1, -1)
    user_distances = cdist(user_vector, tweet_vectors)[0]
    mean_user_distance = np.mean(user_distances)

    # Set up the plot
    plt.figure(figsize=(10, 6), dpi=100)
    ax = sns.histplot(distances, kde=True, color='skyblue', label='Distance Distribution', element='step', stat='density')
    
    # Calculate KDE for distance distribution
    kde = gaussian_kde(distances)
    
    # Get density at mean user distance
    density_at_mean = kde(mean_user_distance)
    
    # Annotate mean user distance on the plot
    plt.annotate('', xy=(mean_user_distance, density_at_mean), xytext=(mean_user_distance, density_at_mean + 0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
    plt.text(mean_user_distance + 0.1, density_at_mean + 0.1, 'Your post', verticalalignment='center', color='red')
    
    # Finalize the plot
    plt.title('Tweet Distance Spectrum')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run the main function and get results
    tweet_embeddings, user_tweet_embedding, platform_tweets = main()
    
    # Plot the distribution of distances
    plot_distance_distribution(tweet_embeddings, user_tweet_embedding)

    # Data storage in JSON format for later analysis
    data = {
        'platform_tweets': platform_tweets, 
        'distances': tweet_embeddings.tolist(), 
        'user_tweet': user_input_tweet, 
        'user_tweet_distance': np.mean(user_tweet_embedding)
    }

    # Write data to a file
    with open('X_API_Sampled_Data.json', 'w') as file:
        json.dump(data, file)
