
import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as components



def clean_context(text:str):
    text = re.sub('nan', "", text)
    text = re.sub('a cap', 'acap', text)
    text = re.sub('\sn\s', ' and ', text)
    text = re.sub(' n ', ' and ', text)
    text = re.sub("'n'", 'and', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text 

def clean_artist_genre(text:str):
    if isinstance(text, str):
        text = ' '.join(text.split(',')).strip()
        text = re.sub('a cap', 'acap', text)
        text = re.sub('hip hop', 'hiphop', text)
        text = re.sub('bomba y plena', 'bomba-y-plena', text)
        text = re.sub('p funk', 'pfunk', text)
        text = re.sub('g funk', 'gfunk', text)
        text = re.sub("'s", '', text)
        text = re.sub("alt z", 'altz', text)
        text = re.sub('album ', '', text)
        text = re.sub("d'a", 'da', text)
        text = re.sub('r&b', 'rnb', text) 
        text = re.sub("'n'", 'and', text)
        text = re.sub('\s\d', '', text) 
        text = re.sub('-', '', text) 
        text = re.sub('\s+', " ", text.strip(string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)
    return text 

def prep_context(data:pd.DataFrame):
    data['context'] = data['context'].map(clean_context) 
    data['genre'] = data['artist_genre'].map(clean_artist_genre)
    data['context'] = data['genre'] + " " + data['context']
    data['context'] = data['context'].str.strip().str.replace('\s+', " ", regex=True)
    # raw['context'] = raw['context'].apply(lambda x: ' '.join(list(set(x.split(" ")))) if isinstance(x, str) else "")
    data['context'] = data['context'].apply(lambda x: ' '.join(list(x.split(" "))) if isinstance(x, str) else "")
    return data

@st.cache
def load_data():
    df = pd.read_csv('../data/cleaned_data.csv')
    df = df[df['artist_genre'].notnull()].reset_index()
    drop_cols = ['artist_id', 'mode', 'key', 'loudness', 'release_date']
    df = df.drop(drop_cols, axis='columns')
    df = prep_context(df) # prep context column for vectorizing
    return df

@st.cache
def load_w2v_data():
    df = pd.read_csv('../data/w2v_vectors.csv') # Prepare in advance for faster loading
    return df


def normalize(data, col, col_name):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data[col])
    df = pd.DataFrame(scaler.fit_transform(df), columns = col_name)
    return df

def get_genres(data:pd.DataFrame, col_name:str):
    df = data[col_name].str.split(',')
    df = df.explode(col_name).drop_duplicates()
    genre_list = np.sort(df.unique()).tolist()
    genre_list.insert(0, 'Select All')
    return genre_list

def get_list(data:pd.DataFrame, col_name:str):
    artist_list = np.sort(data[col_name].unique()).tolist()
    return artist_list

def get_periods(data:pd.DataFrame, col_name:str):
    period_list = data['year'].unique().tolist()
    period_list = sorted(list(set([str(item)[:3] + '0s' for item in period_list])))
    return period_list

def vectorize(data, col):
    vect = TfidfVectorizer(use_idf=True, token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")
    matrix = vect.fit_transform(data[col].values)
    df = pd.DataFrame(matrix.toarray())
    df.columns = ['genre' + "_" + i for i in vect.get_feature_names()]
    return df    
    


def v_spacer(obj, height) -> None:
    for _ in range(height):
            obj.write('\n')

def miniplayer(track_id:str):
    urlstring = "https://open.spotify.com/embed/track/" + track_id
    components.iframe(urlstring, height=80,width=300)

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

st.title('My Music Recommendation App')



raw = load_data() # read_csv
df = raw.copy()

# One hot encoding for popularity
df['popularity_grp5'] = df['popularity'].apply(lambda x: int(x/5))
pop_df = pd.get_dummies(data=df[['popularity_grp5']], columns=['popularity_grp5'], prefix='pop_grp')
df = df.drop(['popularity_grp5'], axis='columns')

# One hot encoding for year / period
df['period'] = df['year'].apply(lambda x: str(int(round(x/5, 0))*5) + 's')
period_df = pd.get_dummies(data=df[['period']], columns=['period'], prefix='period') 
df = df.drop(['period'], axis='columns')

# Min max scaling for audio features
scale_cols = ['acousticness', 'danceability', 'energy','instrumentalness', 'liveness', 'speechiness', 'valence', 'tempo']
new_cols = [col + '_scaled' for col in scale_cols ]
minmax_df = normalize(df, scale_cols, new_cols)

# Vectorizing genre tags using tfidf and word2vec
tfidf_df = vectorize(df, 'context')
w2v_df = load_w2v_data()

temp_df = df.copy()
display_cols = ['name', 'artist_name', 'popularity', 'explicit', 'artist_genre' ]


MIN_YEAR = df['year'].min()
MAX_YEAR = df['year'].max()


# Initialize session_state variables
if 'min_year' not in st.session_state:
    st.session_state['min_year'] = MIN_YEAR
if 'max_year' not in st.session_state:
    st.session_state['max_year'] = MAX_YEAR
if 'explicit' not in st.session_state:
    st.session_state['explicit'] = [0,1]
if 'low_pop' not in st.session_state:
    st.session_state['popu;arity'] = "Select All"
if 'low_valence' not in st.session_state:
    st.session_state['low_valence'] = 0
if 'high_valence' not in st.session_state:
    st.session_state['high_valence'] = 1
if 'genre_value' not in st.session_state:
    st.session_state['genre'] =  "Select All"
if 'artist_value' not in st.session_state:
    st.session_state['artist'] =  "Select All"        
if 'track' not in st.session_state:
    st.session_state['track'] = "" 
if 'vect' not in st.session_state:
    st.session_state['vect'] = "TD-IDF"     

    
with st.container():                         
    ### SIDEBAR CONTEXT ###
    st.sidebar.write('### Search and Filter')

    
    # Period Field
    period_list = ['Select All'] + get_periods(df, 'year')
    period = st.sidebar.selectbox('Select Period', period_list)  
    if period == 'Select All':
        st.session_state['min_year'] = MIN_YEAR
        st.session_state['max_year'] = MAX_YEAR
    else:
        st.session_state['min_year'] = int(period[:-1])
        st.session_state['max_year'] = st.session_state['min_year'] + 9
        temp_df = temp_df[(temp_df['year'] >= st.session_state['min_year']) & (temp_df['year'] <= st.session_state['max_year'])]
    
    # Explicit Language
    explicit_list = ['Select All', 'No', 'Yes']
    explicit = st.sidebar.selectbox("Include explicit language", explicit_list)   
    if explicit == 'Select All':
        st.session_state['explicit'] = [0,1]
    else:    
        if explicit == 'Yes'    :
            st.session_state['explicit'] = [1]
        else:
            st.session_state['explicit'] = [0]
        temp_df = temp_df[(temp_df['explicit'].isin(st.session_state['explicit']))] 

    # Popularity
    pop_list = ['Very popular', 'Popular', 'Neutral', 'Not Popular', 'Very Not Popular']
    pop_dict = dict(zip(pop_list, [(80,100),(60,79),(40,59),(20,39),(0,19)]))
    popularity = st.sidebar.selectbox("Select Popularity", ['Select All'] + pop_list)
    if popularity != "Select All" and not None:
        st.session_state['popularity'] = popularity
        low_pop, high_pop = pop_dict[popularity]
        temp_df = temp_df[(temp_df['popularity'] >= low_pop) & (temp_df['popularity'] <= high_pop)]
    else:
        st.session_state['popularity'] = 'Select All'
   
    
    # Valence / Mood
    mood_list = ['Select All', 'Happy | Positive | Cheerful', 'Sad | Angry']
    mood = st.sidebar.selectbox("Select Mood", mood_list)
    if mood == 'Select All':
        st.session_state['low_valence'] = 0
        st.session_state['high_valence'] = 1
    else:    
        if mood == 'Happy | Positive | Cheerful':
            st.session_state['low_valence'] = 0.5
            temp_df = temp_df[(temp_df['valence'] >= st.session_state['low_valence'])]
    
        else:
            st.session_state['high_valence'] = 0.5
            temp_df = temp_df[(temp_df['valence'] <= st.session_state['high_valence'])]
    
    # Genre Filter
    genre_list = get_genres(temp_df, 'artist_genre')
    genre = st.sidebar.selectbox("Select Genre", genre_list)
    if genre != "Select All" and not None:
        st.session_state['genre'] = genre
        temp_df = temp_df[temp_df['artist_genre'] == st.session_state['genre']]
    else:
        st.session_state['genre'] = 'Select All'
    
        
    # Artist Filter
    artist_list = ['Select All'] + get_list(temp_df, 'artist_name')
    artist = st.sidebar.selectbox("Select Artist", artist_list)
    if artist != "Select All" and not None:
        st.session_state['artist'] = artist
        temp_df = temp_df[temp_df['artist_name']==st.session_state['artist']]
    else:
        st.session_state['artist_value'] = artist
    
    
    st.sidebar.markdown("""---""")




    ### MAIN PAGE CONTENT ###
    st.header('1. Select a song')  
    
    IsTrackSelected = False
    track_list = [''] + get_list(temp_df, 'name')
    track = st.selectbox("Select Track", track_list)
    if track != "" and not None:
        IsTrackSelected = True
        
        # Update and Filter for track
        st.session_state['track'] = track
        temp_df = temp_df[temp_df['name'] == st.session_state['track']]
        if len(temp_df) == 1:
            idx = temp_df.index[0]    
            track_id = df.loc[idx, 'id']
            artist = df.loc[idx, 'artist_name']
            html_string = '<iframe src="https://open.spotify.com/embed/track/' + track_id + \
                '" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
        
        # User Feedback
        st.subheader('Selected: ' + track + ' by ' + artist + " [" + str(idx) + "]")
        
        # Display Preview of selected track
        col1, col2 = st.columns([1, 3])
        components.html( html_string,height=380)  
        #col2.write('Song Stats')    
        v_spacer(obj=st,height=1)
        
    else:
        IsTrackSelected = False
        st.session_state['track'] = track
        
        # Show search results
        st.subheader('Search Results - ' + str(temp_df.shape[0]) + ' items')
        st.write(temp_df[display_cols]) # Displaying the dataframe.    
        v_spacer(obj=st,height=10)

    st.markdown("""---""")
    
    

   
with st.container():            
    # Initialize variable
    trigger = False
    
    ### SIDEBAR CONTENT ###
    v_spacer(obj=st.sidebar,height=5)
    st.sidebar.write('### Recommendation Settings')
    
    # No. of Recommendations
    num_songs = st.sidebar.number_input('Number of Recommendations', min_value=0, value=20)
    
    # Vectorizer Type: TFIDF vs W2V
    vect = st.sidebar.radio('Text Processing', ['TD-IDF', 'Word2Vec'])
    st.session_state['vect'] = vect  
    
    
    
    ### MAIN PAGE CONTENT ###
    st.header('2. Generate Recommendations')  
    
    # Show button only when track has been selected
    if IsTrackSelected: 
        trigger = st.button('Recommend Songs')
    else:
        st.write('### No Song was selected. Please select a song first.')    
    

    if trigger:   # returns True / False
        
        # Calculate Cosine Similarity scores    
        df['sim_score_pop'] =cosine_similarity(pop_df.values, pop_df.iloc[idx].values.reshape(1, -1))[:,0]
        df['sim_score_year'] = cosine_similarity(period_df.values, period_df.iloc[idx].values.reshape(1, -1))[:,0]
        df['sim_score_audio'] = cosine_similarity(minmax_df.values, minmax_df.iloc[idx].values.reshape(1, -1))[:,0] 
        
        if st.session_state['vect'] == "TD-IDF":
            df['sim_score_genre'] = cosine_similarity(tfidf_df.values, tfidf_df.iloc[idx].values.reshape(1, -1))[:,0]
        elif st.session_state['vect'] == "Word2Vec":
            df['sim_score_genre'] = cosine_similarity(w2v_df.values, w2v_df.iloc[idx].values.reshape(1, -1))[:,0]
        
        # Factor weightages
        df['sim_score_audio'] = (0.60 * df['sim_score_audio']) 
        df['sim_score_genre'] = (0.30 * df['sim_score_genre']) 
        df['sim_score_year'] =  (0.05* df['sim_score_year']) 
        df['sim_score_pop'] = (0.05 * df['sim_score_pop'])
        
        # Consolidate Final Score
        df['sim_score'] = df['sim_score_audio'] + df['sim_score_genre'] + df['sim_score_year'] + df['sim_score_pop']  
        
        # Sort Values by score
        results = df.sort_values('sim_score',ascending = False).head(num_songs+1)
        
        with st.expander('Recommendation Details'):
            show_cols = ['name', 'artist_name','context', 'sim_score', 'sim_score_audio', 'sim_score_genre', 'sim_score_year', 'sim_score_pop'] #+ display_cols
            st.write(results[show_cols]) # Displaying the dataframe.
    
        # Generate Recommendation (includes miniplayer)
        for row, data in results.reset_index().iterrows():
            track_name = data['name']
            artist_name = data['artist_name']
            if (track_name, artist_name) != (track, artist):
                PrntString = str(row) +  '. ' + track_name + " by " + artist_name
                st.write(PrntString)
                miniplayer(data['id'])
        
    else:
        st.write('')


# with st.container():  
#     with st.expander('Session Stats'):
#         st.write( st.session_state)

