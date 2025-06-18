import glob
import json
import pickle
import re
import sqlite3
import pandas as pd
import requests
# from apikeys import steamCreds, sqlpw  # * local dependancy
from bs4 import BeautifulSoup
from steam import Steam
from tqdm import tqdm

conn = sqlite3.connect('ytbrdb')
c = conn.cursor()

pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 30)
pd.set_option('display.max_columns', 48)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
pd.set_option('compute.use_numba', True)
pd.set_option('display.date_yearfirst', True)
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.width', None)
pd.set_option('plotting.backend', 'plotly')
pd.set_option("display.show_dimensions", True)
pd.set_option("styler.latex.hrules", True)
s = Steam(steamCreds)


def updategamesdb():
    df_vids = pd.concat([pd.read_pickle(i[2:]) for i in glob.glob(
        './pickles/individual/*full.pkl')], ignore_index=True)  # * local
    df_vids.dropna(inplace=True)
    # minimum number of videos before a game is considered
    df_vids = df_vids.groupby("game").filter(lambda x: len(x) >= 10)

    df_games = pd.DataFrame(df_vids.game.value_counts()).reset_index()
    del df_vids
    df_games.drop_duplicates(subset=['index'], inplace=True)
    df_games = df_games[~df_games.index.isin(['1996', '2000', '2004', '2009', '2010', '2011', '2012', '2013',
                                             '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', 'VRChat', 'Youtubers Life'])]
    df_games.rename(columns={'index': 'name', 'game': 'count'}, inplace=True)
    df_games.set_index('name', inplace=True)

    def returnData(data: dict, steam: Steam) -> dict:
        """
        Extracts a dictionary of relevant data from a game's json object

        Args:
            data (dict): dictionary of steam <data> json object for a single game
            steam (Steam): steam api instance initialized with key

        Returns:
            dict: dictionary of relevant information for the game
        """
        container = {
            'appid': ['steam_appid'],
            'requiredAge': ['required_age'],
            'metacritic': ['metacritic', 'score'],
            'price': ['price_overview', 'initial'],
            'windows': ['platforms', 'windows'],
            'mac': ['platforms', 'mac'],
            'linux': ['platforms', 'linux'],
            'releaseDate': ['release_date', 'date'],
            'acheiveCount': ['achievements', 'total'],
            'recs': ['recommendations', 'total'],
            'publishers': ['publishers'],
            'developers': ['developers'],
            'desc': ['about_the_game'],
            'header_image': ['header_image'],
            'genres': None,
            'categories': None
        }
        for key in container:
            if key in ['genres', 'categories']:  # multiple responses
                try:
                    _cont = []
                    for i in data[key]:
                        _cont.append(i['description'])
                    container[key] = _cont[:5]
                    continue
                except BaseException:  # ! missing data, silently continuing
                    container[key] = None
                    continue
            try:
                _ = data.copy()
                for i in container[key]:
                    _ = _[i]
                container[key] = _
            except KeyError:  # ! missing data, silently continuing
                container[key] = None
                continue
            except AttributeError:  # ! missing data, silently continuing
                container[key] = None
                continue
        try:
            container['publishers'] = container['publishers'][:3]
            container['developers'] = container['developers'][:3]
        except BaseException:  # TODO if either is missing; find what error it throws
            pass
        return container

    def gameSearch(gamename: str, steam: Steam) -> dict:
        """
        Searches for a game's app_id, returns it's 'data' json object

        Args:
            gamename (str): Title of game
            steam (Steam): steam api instance initialized with key

        Raises:
            NameError: If game cannot be found

        Returns:
            dict: json object of the game's 'data'
        """
        # getting the appid
        # * Attempt 1: Steamspy
        try:
            steamspyids = pd.read_pickle('./pickles/appid.pkl')
        except FileNotFoundError:
            steamspy = pd.DataFrame.from_dict(requests.get(
                url="https://steamspy.com/api.php", params={"request": "all"}).json(), orient='index')
            steamspyids = steamspy[['appid', 'name']].sort_values(
                'appid').reset_index(drop=True)
            steamspyids.to_pickle('./pickles/appid.pkl')

        _ = steamspyids[steamspyids.name.isin([gamename])].appid.to_list()
        if _:
            possibleids = _
        else:
            # * Attempt 2: python.steam.api
            try:
                possibleids = [app['id'] for app in steam.apps.search_games(gamename)[
                    'apps']]
        # * Attempt 3: steam search suggestions
            except BaseException:
                searchapi = "https://store.steampowered.com/search/suggest"
                with requests.Session() as session:
                    # ! something is wrong, occasionally returns nonenglish pages
                    params = {"l": "english", "term": gamename,
                              "category1": "998", "cc": "US"}
                    response = session.get(searchapi, params=params)
                    response.raise_for_status()
                    result = BeautifulSoup(response.text, "html.parser").find(
                        'a')
                    if result:
                        bundle_id = result.get("data-ds-bundleid")
                        app_id = result.get("data-ds-appid")

                        if bundle_id:
                            # name = result.find(
                            # "div", class_="match_name").get_text()
                            bundle_data = json.loads(
                                re.sub(
                                    r"&quot;", '"', result["data-ds-bundle-data"]
                                )
                            )
                            possibleids = [app_id for item in bundle_data["m_rgItems"]
                                           for app_id in item["m_rgIncludedAppIDs"]]
                        elif app_id:
                            possibleids = [app_id]
                        else:
                            raise NameError
                    else:
                        raise NameError
        if not possibleids:
            raise NameError  # ! Game not found

        finaldata = {}
        for i in range(len(possibleids)):
            idguess = possibleids[i]
            try:
                data = json.loads(steam.apps.get_app_details(idguess))[
                    str(idguess)]['data']
                if i == 0:  # return first candidate as fallback
                    finaldata = data.copy()
                if data['type'] == 'game':
                    finaldata = data.copy()
                    break
            except BaseException:
                continue
        if not finaldata:
            raise NameError  # ! Data not found
        return finaldata

    def steamPipeline(df: pd.DataFrame, steam: Steam) -> pd.DataFrame:
        """
        Pipeline to iterate through a dataframe and populate columns. Updates pickled dataframe if available

        Args:
            df (pd.DataFrame): dataframe with game names # ! at column index 0
            steam (Steam): steam api instance initialized with key

        Returns:
            pd.DataFrame: Original dataframe with new columns, missing data silently filled with nan
        """
        c.execute('CREATE TABLE IF NOT EXISTS youtubers (appid text, requiredAge number, metacritic number, price number, windows number, mac number, linux number, releaseDate text, acheiveCount text, recs, number, publishers text, developers text, descr text, header_image text, genres text, categories text)')
        conn.commit()
        df = df.reindex(
            columns=df.columns.tolist() + ['appid', 'requiredAge', 'metacritic', 'price', 'windows', 'mac', 'linux', 'releaseDate', 'acheiveCount', 'recs', 'publishers', 'developers', 'desc', 'header_image', 'genres', 'categories'])
        try:
            df_old = pd.read_pickle('./pickles/df_games.pkl')
            df_old.set_index('name', inplace=True)
            df = df.fillna(df_old)
        except FileNotFoundError:  # first run
            pass
        try:
            nonsteam = pickle.load(open('./pickles/nonsteamlist.pkl', 'rb'))
        except FileNotFoundError:  # first run
            nonsteam = []
        t = tqdm(df.itertuples(), total=df.shape[0])
        errors = 0
        for row in t:
            name = row[0]
            # skipping scraped rows
            if (df.loc[name, :].isna().sum() <
                    16):  # TODO change to a lower number later (~8?)
                continue
            elif name in nonsteam:
                continue
            try:
                data = gameSearch(name, steam)
            except NameError:  # ! Game cannot be found, continuing
                errors += 1
                t.set_postfix(
                    {'Total errors': errors, "Last missing game": name})
                continue
            result = returnData(data, steam)
            for column in result:
                if isinstance(result[column], list):
                    df[column] = df[column].astype('object')
                df.at[name, column] = result[column]
        nonsteam = 0
        filter = df.isna().sum(axis=1) == 16  # empty games
        nonsteam = df[filter].index.to_list()  # filter so don't search again
        df = df[~filter]
        df.reset_index(inplace=True)
        # df.to_sql('youtubers', conn, if_exists='replace', index = False)
        pickle.dump(nonsteam, open('./pickles/nonsteamlist.pkl', 'wb'))
        df.to_pickle('./pickles/df_games.pkl')
        return df

    df_games = steamPipeline(df_games, s)
