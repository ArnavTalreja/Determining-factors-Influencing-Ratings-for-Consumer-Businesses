#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import gzip
import json

url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-Hawaii_10.json.gz"

dataset = []

response = requests.get(url, stream=True)
with gzip.GzipFile(fileobj=response.raw) as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)  

print(f"Total records: {len(dataset)}")


# In[2]:


url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-Hawaii.json.gz"

metadata = []

response = requests.get(url, stream=True)
with gzip.GzipFile(fileobj=response.raw) as f:
    for line in f:
        data = json.loads(line)
        metadata.append(data)  

print(f"Total records: {len(metadata)}")


# In[3]:


dataset[0:5]


# In[4]:


import pandas as pd
from datetime import datetime

review_df = pd.DataFrame(dataset)

review_df['time'] = review_df['time'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))

review_df['text'] = review_df['text'].fillna('No Comment')
review_df['pics'] = review_df['pics'].fillna('No Pics')
review_df['resp'] = review_df['resp'].fillna('No Response')

review_df.drop_duplicates(subset=['user_id', 'gmap_id'], inplace=True)

review_df = review_df[review_df['rating'].between(1, 5)]

review_df.sort_values(by='time', inplace=True)

print(f"Cleaned dataset contains {len(review_df)} records.")
review_df.head()


# In[5]:


metadata[0:5]


# In[6]:


cleaned_data = []

for entry in metadata:
    misc = entry.get("MISC", {}) or {} 
    cleaned_entry = {
        "name": entry.get("name", "N/A"),
        "address": entry.get("address", "N/A"),
        "gmap_id": entry.get("gmap_id", "N/A"),
        "description": entry.get("description") or "N/A",
        "latitude": round(entry.get("latitude", 0.0), 6),
        "longitude": round(entry.get("longitude", 0.0), 6),
        "category": ', '.join(entry.get("category", []) or []),  
        "avg_rating": entry.get("avg_rating", "N/A"),
        "num_of_reviews": entry.get("num_of_reviews", 0),
        "price": entry.get("price", "N/A"),
        "hours": {day: time for day, time in (entry.get("hours", []) or [])},
        "service_options": ', '.join(misc.get("Service options", [])),
        "accessibility": ', '.join(misc.get("Accessibility", [])),
        "popular_for": ', '.join(misc.get("Popular for", [])),
        "offerings": ', '.join(misc.get("Offerings", [])),
        "atmosphere": ', '.join(misc.get("Atmosphere", [])),
        "state": entry.get("state", "N/A"),
        "url": entry.get("url", "N/A")
    }
    cleaned_data.append(cleaned_entry)

meta_df = pd.DataFrame(cleaned_data)

meta_df.head()


# In[7]:


import math
from collections import defaultdict

def compute_tf_idf(dataframe, column="text"):
    """
    Compute TF-IDF for the specified column in the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing the text data.
        column (str): The column on which to compute TF-IDF. Default is "text".

    Returns:
        dict: A dictionary where keys are document indices and values are dictionaries of word TF-IDF scores.
    """
    tokenized_text = dataframe[column].fillna("").apply(lambda x: x.split())

    tf = []
    for doc in tokenized_text:
        word_count = defaultdict(int)
        for word in doc:
            word_count[word] += 1
        doc_tf = {word: count / len(doc) for word, count in word_count.items()}
        tf.append(doc_tf)

    df = defaultdict(int)
    total_docs = len(tokenized_text)
    for doc in tokenized_text:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1

    idf = {word: math.log(total_docs / (1 + freq)) for word, freq in df.items()}

    tf_idf = {}
    for idx, doc_tf in enumerate(tf):
        doc_tf_idf = {word: tf_value * idf[word] for word, tf_value in doc_tf.items()}
        tf_idf[idx] = doc_tf_idf

    return tf_idf


tf_idf_scores = compute_tf_idf(review_df, column="text")

print("TF-IDF Scores for the first document:")
print(tf_idf_scores[0])


# In[9]:


merged_df = pd.merge(review_df, meta_df, on='gmap_id', how='left')
merged_df.head()


# In[32]:


# extracting parameters


# In[10]:


#reviews per user

from collections import defaultdict

reviewsPerUser = defaultdict(list)

for idx, row in merged_df.iterrows():
    user_id = row['user_id']
    review_text = row['text']
    # Get the corresponding TF-IDF scores for this review
    review_tfidf = tf_idf_scores.get(idx, {})
    
    # Create a dictionary of review information and append it to the reviewsPerUser dict
    review_info = {
        'review_text': review_text,
        'tf_idf': review_tfidf
    }
    reviewsPerUser[user_id].append(review_info)



# In[19]:


print(dict(list(reviewsPerUser.items())[:1]))


# In[21]:


# reviews per place

reviewsPerPlace = defaultdict(list)

for idx, row in merged_df.iterrows():
    place_name = row['name_y']
    review_text = row['text']
    review_tfidf = tf_idf_scores.get(idx, {})
    
    review_info = {
        'review_text': review_text,
        'tf_idf': review_tfidf
    }
    reviewsPerPlace[place_name].append(review_info)


# In[22]:


# place per user

placePerUser = defaultdict(list)

for idx, row in merged_df.iterrows():
    user_id = row['user_id']
    place_name = row['name_y']
    review_text = row['text']
    review_tfidf = tf_idf_scores.get(idx, {})
    
    review_info = {
        'place_name': place_name,
        'tf_idf': review_tfidf
    }
    placePerUser[user_id].append(review_info)



# In[23]:


# users per place

usersPerPlace = defaultdict(list)

for idx, row in merged_df.iterrows():
    place_name = row['name_y']
    user_id = row['user_id']
    review_text = row['text']
    review_tfidf = tf_idf_scores.get(idx, {})
    
    review_info = {
        'user_id': user_id,
        'tf_idf': review_tfidf
    }
    usersPerPlace[place_name].append(review_info)




# In[24]:


# metadata per place

metadataPerPlace = {}

# Populate metadataPerPlace with the relevant metadata for each place
for idx, row in merged_df.iterrows():
    place_name = row['name_y']
    metadata = {
        'price': row['price'],
        'hours': row['hours'],
        'service_options': row['service_options'],
        'accessibility': row['accessibility'],
        'popular_for': row['popular_for'],
        'offerings': row['offerings'],
        'atmosphere': row['atmosphere'],
        'state': row['state'],
        'url': row['url']
    }
    metadataPerPlace[place_name] = metadata




# In[25]:


# EDA


# In[26]:


print(meta_df[['avg_rating', 'num_of_reviews']].describe())


# In[31]:


import folium
from folium.plugins import MarkerCluster

# Create the base map (California coordinates as central point)
map_center = [20.7967, -156.3319]  # Central coordinates for Hawaii

m = folium.Map(location=map_center, zoom_start=6)

# Create a marker cluster to group nearby points
marker_cluster = MarkerCluster().add_to(m)

# Loop through your metadata DataFrame and add a marker for each company
for _, row in meta_df.iterrows():
    # Extract necessary data
    name = row['name']
    latitude = row['latitude']
    longitude = row['longitude']
    avg_rating = row['avg_rating']
    
    # Set color based on the rating (you can use a gradient or specific color thresholds)
    if avg_rating >= 4.5:
        color = 'green'
    elif avg_rating >= 3.5:
        color = 'orange'
    else:
        color = 'red'

    # Add a marker with the rating and other details
    folium.Marker(
        location=[latitude, longitude],
        popup=f"<strong>{name}</strong><br>Rating: {avg_rating}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map in the notebook
m



# In[ ]:




