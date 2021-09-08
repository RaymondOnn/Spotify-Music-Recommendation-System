# Project Capstone: Spotify Content-based Recommendation System

##  Executive Summary
On any given day, the average individual makes a range of conscious decisions about their media consumption. As we navigate through these countless choices, recommendation system algorithms such as collaborative filtering, utilize historical behavioral data patterns, to allow users to find relevant and enjoyable content, thereby increasing user satisfaction and driving engagement for a given product or service. 

However, in some cases, prior behavioral or user data is not available and collaborative filtering methods are no longer viable. Such situations are common in the early phases of startups in the media streaming space. With this in mind, I sought to build a content-based recommendation system and explored with different NLP techniques such as TD-IDF and Word2Vec to help enhance its performance

##  Problem Statement
This project seeks to explore better ways of boosting music discovery especially for brand new artists or old or unpopular music, given the absence of prior behavioral or user data.
Such recommendation systems will be useful in the early phases of a startup when it is still establishing its own user base. 

## Technical Goals
1. Learn to use Airflow and Docker in the data collection phase
2. Familiarise oneself with Google Cloud API
3. Explore the different ways content-based recommendation systems can be built
4. Learn to implement Word2Vec embeddings
5. Develop a demo app using Streamlit
 
## Dataset
* The dataset consists of the songs released between 1930 and 2020  (1000 tracks per year).
* This data collection process was triggered automatically every 5 mins using **Airflow**, which was running in a **Docker** container
* In each cycle, attributes of songs released in a particular year was scraped from Spotify'S Search API via the spotipy library.
* After some initial data transformation, the data was stored as csv files in **Google Cloud Storage** and then loaded into **BigQuery**
* After data cleaning, we are left with 74.030 rows

### Data Dictionary
* Dataset consists of 21 columns

|Feature|Type|Description|
|---|---|---|
|artist_id|text| The id of the artist| 
|artist_name|text| The name of the artist|
|id|text| The id of the track| 
|name|text|  The name of the track| 
|release_date|text| The release date of the track| 
|year|int| The release year of the track| 
|duration_ms|int|The duration of the track in milliseconds. | 
|explicit|int| Indicates the use of explicit language in the track| 
|popularity|int| The popularity of the track. Ranges from 0 to 100| 
|artist_genre|text| The genre of the track| 
|danceability|float| A measure of how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. Ranges from 0 to 1| 
|energy|float| A perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Ranges from 0 to 1| 
|key|int| The estimated overall key of the track. Integers map to pitches using standard [Pitch Class notation](https://en.wikipedia.org/wiki/Pitch_class) . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.| 
|loudness|float| The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks.| 
|mode|int| Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.| 
|speechiness|float| A measure of the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.  Ranges from 0 to 1| 
|acousticness|float| A confidence measure from 0.0 to 1.0 of whether the track is acoustic.| 
|instrumentalness|float| A measure of whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. Ranges from 0 to 1| 
|liveness|float| A measure of the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. Ranges from 0 to 1| 
|valence|float| the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).| 
|tempo|float| The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece, and derives directly from the average beat duration.|  



##  Preprocessing

Before calculating similarity scores, here are the steps done to preprocess the data:

-   Dropping of unnecessary features
-   Feature engineering i.e. grouping of years in groups of 5
-   Normalization (Minmax Scaling) on float features
-   One hot Encoding for popularity and period
- With the 'genre' column, 2 different vectorization methods were used, **TF-IDF** and **Word2Vec**

Multiple sets of similarity scores were then calculated separately using cosine similarity and then consolidated using the weighted approach


## Conclusion & Next Steps
Content-based recommendation using Spotify track attributes is generally effective at creating good song recommendations and increasing music discovery

However, there is definitely scope to improve its performance:
1. **Greater volume of  data**: Larger song database would allows the recommendation system to pick and choose songs that are more similar.
2.  **Different types of data**: With playlist data, we can use song embeddings to train recommendation systems to understand context when recommending songs, For example, to make a "soft pop rock" playlist
3. **User Data**: Incorporate user data such as prior usage patterns and user feedback would help customize recommendations to suit user preferences better