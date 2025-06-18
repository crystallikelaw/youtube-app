import glob
import re

# from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# from tqdm import tqdm


def cleandata():
    df_vids = pd.concat([pd.read_pickle(i) for i in glob.glob(
        './individual/*full.pkl')], ignore_index=True)  # * local
    df_vids.dropna(inplace=True)
    # minimum number of videos before a game is considered
    df_vids = df_vids.groupby("game").filter(lambda x: len(x) >= 5)
    df_vids = df_vids[['channel', 'title', 'date', 'likes', 'comments', 'views',
                       'game']].copy()
    df_vids.rename(columns={'channel': 'Channel Title', 'title': 'Video Title', 'date': 'Upload Date', 'likes': 'Likes', 'comments': 'Comments', 'views': 'Views',
                            'game': 'Game Played'}, inplace=True)
    df_vids = df_vids[~df_vids['Game Played'].isin(['1996', '2000', '2004', '2009', '2010', '2011', '2012', '2013',
                                                    '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', 'VRChat', 'Youtubers Life'])].copy()
    df_vids = df_vids.astype(
        {'Likes': 'int', 'Comments': 'int', 'Views': 'int'})
    df_vids['Engagement'] = df_vids['Likes'] + df_vids['Comments']
    df_vids.drop(columns=['Likes', 'Comments'])
    df_vids['Upload Date'] = pd.to_datetime(
        df_vids['Upload Date'], errors='coerce').dt.tz_localize(None)
    df_vids.replace('SeaNanners Gaming Channel', 'SeaNanners', inplace=True)
    df_vids.replace('Splattercatgaming', 'Splattercat', inplace=True)

    df_games = pd.read_pickle('../pickles/df_games.pkl')
    df_games = df_games[['name', 'count', 'metacritic', 'price',
                         'releaseDate', 'desc', 'header_image', 'genres',
                         'categories']].copy()
    df_games.rename(columns={'name': 'Game Title', 'count': 'Video Count', 'metacritic': 'Metacritic Score', 'price': 'Price (USD)',
                             'releaseDate': 'Release date (steam)', 'desc': 'Description', 'genres': 'Genres',
                             'categories': 'Tags'}, inplace=True)
    df_games['Price (USD)'] = df_games['Price (USD)'] / 100
    df_games['Release date (steam)'] = pd.to_datetime(
        df_games['Release date (steam)'], errors='coerce')
    df_games['Price (USD)'][df_games['Price (USD)'] >= 120] = np.nan
    df_games['Description'] = df_games['Description'].apply(fullClean)

    df_games.to_pickle('./dfgames.p')

    df_vids = df_vids.merge(df_games[['Game Title', 'Genres']],
                            left_on="Game Played", right_on="Game Title", how='inner')
    df_vids.drop(columns=['Game Title'], inplace=True)
    df_vids['Upload Year'] = df_vids['Upload Date'].dt.year
    df_vids.to_pickle('./dfvids.p')
    return 0


2 + 3


# Cleaning functions
def cleanHtml(text: str) -> str:
    """
    Removes HTML symbols from text

    Args:
        text (str): input text

    Returns:
        str: text without things like %2E in it
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, r' ', text)


def cleanUrl(text: str) -> str:
    """
    Removes http(s) urls from text

    Args:
        text (str): input text

    Returns:
        str: output text without urls like https://google.com/
    """
    return re.sub(r"https?://\S+|www\.\S+", r' ', text)


def cleanAscii(text: str) -> str:
    """
    Remove non ASCII characters

    Args:
        text (str): input text

    Returns:
        str: text without things like carriage returns, tabs, null bytes
             removes from \x00 (null byte) to \x7f (delete)
    """
    return re.sub(r'[^\x00-\x7f]', r' ', text)


def cleanSpecChar(text: str) -> str:
    """
    Removes misc special unicode chars

    Args:
        text (str): input text

    Returns:
        str: text without emojis, pictographs, maps, flags..
    """
    specPattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return specPattern.sub(r' ', text)


def cleanSpaces(text: str) -> str:
    """
    Converts multiple spaces in text to a single space

    Args:
        text (str): input text

    Returns:
        str: text with at most contiguous space
    """
    return re.sub(r'\s{2,}', r' ', text)


def fullClean(text: str) -> str:
    """
    Compiles the above cleaning functions

    Args:
        text (str): input text

    Returns:
        str: cleaned output text, lowercase
    """
    return cleanSpaces(cleanSpecChar(
        cleanAscii(cleanHtml(cleanUrl(text))))).strip()
