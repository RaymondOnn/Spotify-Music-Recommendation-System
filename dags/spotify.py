import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.contrib.hooks.bigquery_hook import BigQueryHook

CLIENT_ID = '6fe3c874d63e409788801c7056db56f3'
CLIENT_SECRET = 'be2c76aa3c8f4e269ccc982a1b81b112'
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

DATASET_ID = 'tracks'
BUCKET_NAME = "spotify_dsi23_capstone"
GOOGLE_CLOUD_CONN_ID = 'gcs_default'
BIGQUERY_TABLE_NAME = "tblTracks"
CSV_ENCODING = 'utf-8-sig'

def get_min_year():
    query = '''
    SELECT 
        MIN(year) 
    FROM 
        [spotify-324112.tracks.tblTracks]'''
    bq = BigQueryHook(bigquery_conn_id=GOOGLE_CLOUD_CONN_ID, delegate_to=None, use_legacy_sql=True)
    min_year = bq.get_pandas_df(query, dialect='legacy').values.tolist()[0][0]
 
    return min_year


YEAR = get_min_year() - 1
GCS_OBJECT_NAME = str(YEAR) + ' Spotify Track Data.csv'

def get_track_data(year:int):
    offsets = [page * 50 for page in range(20)] # 50 search results per pages. Given total of 1000 results per search, we have 20 pages
    df_list = []
    q = 'year:"' + str(year) + '"'
    for offset in offsets:

        lst = []
        response = sp.search(q, limit=50, offset=offset, type='track', market='SG')

        for idx in range(len(response['tracks']['items'])):
            artist_id = response['tracks']['items'][idx]['artists'][0]['id']
            artist_name = response['tracks']['items'][idx]['artists'][0]['name']
            track_id = response['tracks']['items'][idx]['id'] 
            track_name = response['tracks']['items'][idx]['name'] 
            track_release_date = response['tracks']['items'][idx]['album']['release_date'] 
            track_duration_ms = response['tracks']['items'][idx]['duration_ms'] 
            track_explicit = int(response['tracks']['items'][idx]['explicit']) 
            track_popularity = response['tracks']['items'][idx]['popularity']
            lst.append([artist_id, artist_name, track_id, track_name, track_release_date, year, track_duration_ms, track_explicit, track_popularity]) 
            cols = ['artist_id', 'artist_name', 'id', 'name', 'release_date', 'year', 'duration_ms', 'explicit', 'popularity']

        data = pd.DataFrame(lst, columns=cols)
        df_list.append(data)
    df = pd.concat(df_list)    
    return df

def list_chunks(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

def get_artist_genre(df):    
    temp_df = df[['artist_id']].drop_duplicates()
    df_list = []
    for lst in list_chunks(temp_df['artist_id'].values.tolist(), 50):
        response = sp.artists(lst)['artists']
        id_list = lst
        genre_list = [ dictionary['genres'] for dictionary in response ]

        data = pd.DataFrame(list(dict(zip(id_list, genre_list)).items()), columns=['artist_id', 'artist_genre'])
        df_list.append(data)
    artist_df = pd.concat(df_list) 
    artist_df['artist_genre'] = artist_df['artist_genre'].apply(lambda x: ','.join(x))
    return artist_df

def get_audio_features(df):    
    temp_df = df[['id']].drop_duplicates()
    df_list = []
    for list_chunk in list_chunks(temp_df['id'].values.tolist(), 100):
        lst = []
        response = sp.audio_features(list_chunk)
        
        for idx in range(len(response)):
            track_id = list_chunk[idx]
            try:
                danceability = response[idx]['danceability']
                energy = response[idx]['energy']
                key = int(response[idx]['key']) 
                loudness = response[idx]['loudness']
                mode = int(response[idx]['mode'])
                speechiness = response[idx]['speechiness']
                acousticness = response[idx]['acousticness']
                instrumentalness = response[idx]['instrumentalness']
                liveness = response[idx]['liveness']
                valence = response[idx]['valence']
                tempo = response[idx]['tempo']
                lst.append([track_id, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo]) 
            except:
                song_name = df.loc[df['id'] == track_id, 'name'].values[0]
                song_artist = df.loc[df['id'] == track_id, 'artist_name'].values[0]
                print(f'No audio features available for: {song_name} BY {song_artist}')
            
        cols = ['id','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        data = pd.DataFrame(lst, columns=cols)
        df_list.append(data)
    df = pd.concat(df_list)    
    return df

def merge_df(track_df, artist_df, audio_df):
    df = track_df.merge(artist_df, how='left', on='artist_id')
    df = df.merge(audio_df, how='left', on='id')
    #df['genre'] = df['artist_genre'].apply(lambda x: ','.join(x.split(', ')))
    return df

def get_track_info_year(year:int):
    track_df = get_track_data(year)
    artist_df = get_artist_genre(track_df)
    audio_df = get_audio_features(track_df)
    df = merge_df(track_df, artist_df, audio_df)
    df['key'] = df['key'].fillna(0).astype('int64')
    df['mode'] = df['mode'].fillna(0).astype('int64')
    return df

def save_csv_to_gcs(df, dest_file_name):        
    """Uploads a file to the bucket."""
    
    df.to_csv('temp.csv', index=False, encoding=CSV_ENCODING)
    gcs = GCSHook(gcp_conn_id =GOOGLE_CLOUD_CONN_ID)
  
    #gcs_client = storage.Client(project = gcs._get_field("project"), credentials = gcs._get_credentials())
    destination_blob_name = dest_file_name
    try:
        gcs.upload(bucket_name =BUCKET_NAME,
             object_name = destination_blob_name,
             filename = 'temp.csv')
        print(f"Data saved and uploaded to {destination_blob_name}.")
    except Exception as e:
        print(e)        

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'start_date': datetime(2021, 8, 26)
}



dag = DAG(
    dag_id='spotipy_scraper',
    default_args=default_args,
    catchup=False,
    schedule_interval="*/5 * * * *" # run every 5 mins
)


def has_scraping_completed(**context):
    print('minimum year value from BigQuery: {}'.format(YEAR + 1))
    if  YEAR < 1930: # return bool value from the query
        return 'do_nothing'
    else:
        return 'scrape_data'
    print(f'proceeding to scrape daya for year {YEAR}')
    context['ti'].xcom_push(key='year_to_scrape', value=YEAR)

check_to_start_scraping = BranchPythonOperator(
    task_id='check_to_start_scraping',
    python_callable=has_scraping_completed,
    provide_context=True,
    dag=dag,
    depends_on_past=False
)


def scrape_data(**context):
    ti = context['ti']
    received = ti.xcom_pull(key='year_to_scrape', task_ids='check_to_start_scraping')
    print(f'Received xcom value: {received}')
    print(f'scraping data for year {YEAR}')
    df = get_track_info_year(YEAR)
    save_csv_to_gcs(df, GCS_OBJECT_NAME)    
    print(f'Uploaded {GCS_OBJECT_NAME} to Google Cloud Storage...')



scrape_and_save_data = PythonOperator(
    task_id='scrape_data',
    python_callable=scrape_data,
    dag=dag)


GCS_to_BigQuery = GCSToBigQueryOperator(
        task_id='GCS_to_BigQuery',
        bigquery_conn_id=GOOGLE_CLOUD_CONN_ID,
        bucket=BUCKET_NAME,
        source_objects=[GCS_OBJECT_NAME],
        source_format='CSV',
        skip_leading_rows = 1, 
        field_delimiter= ',',
        destination_project_dataset_table=f"{DATASET_ID}.{BIGQUERY_TABLE_NAME}",
        schema_fields=[ #based on https://cloud.google.com/bigquery/docs/schemas
            {'name': 'artist_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'artist_name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'id', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'release_date', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'year', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'duration_ms', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'explicit', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'popularity', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'artist_genre', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'danceability', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'energy', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'key', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'loudness', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'mode', 'type': 'INT64', 'mode': 'NULLABLE'},
            {'name': 'speechiness', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'acousticness', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'instrumentalness', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'liveness', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'valence', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'tempo', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
        ], 
        autodetect=False,
        create_disposition='CREATE_IF_NEEDED',
        write_disposition='WRITE_APPEND', 
        encoding='utf-8',
        dag=dag
    )

def scraping_complete():
    print('Scraping has completed')


do_nothing = PythonOperator(
    task_id='do_nothing',
    python_callable=scraping_complete,
    do_xcom_push = False,
    dag=dag
)

check_to_start_scraping >> [scrape_and_save_data, do_nothing]
scrape_and_save_data >> GCS_to_BigQuery 
