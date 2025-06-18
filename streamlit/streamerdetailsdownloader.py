# from selenium.webdriver.chrome.service import Service
import os
import re
# from pprint import PrettyPrinter
from time import sleep

import pandas as pd
from apiclient.discovery import build
from func_timeout import FunctionTimedOut, func_timeout
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from tqdm import tqdm

from apikeys import youtubeCreds
import streamlist as st


@st.experimental_singleton
def installff():
    os.system('sbase install geckodriver')
    os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')


_ = installff()


# tqdm.pandas()
# pp = PrettyPrinter().pprint
y = build('youtube', 'v3', developerKey=youtubeCreds)


def startDriver() -> webdriver:
    """
    Starts webdriver instance (machine specific)

    Returns:
        WebDriver: selenium Webdriver
    """
    options = FirefoxOptions()
    options.add_argument("--headless")

    # options = webdriver.ChromeOptions()
    # options.binary_location = '''C://Users//crystallikelaw//AppData//Local//Chromium//Application//chrome.exe'''

    # options.add_argument("--headless")
    options.add_argument('--lang=en')
    # options.add_argument(
    # "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36")
    # service = Service('chromedriver.exe')

    driver = webdriver.Firefox(options=options)
    # driver = webdriver.Chrome(service=service, options=options)
    return driver


def formatNumber(numberStr: str) -> int:
    """
    Formats a str such as '1,245.6K' as an int 1245600
    * removes commas
    * replaces K, B, M
    ! expects input to be a whole number


    Args:
        numberStr (str): number as string

    Returns:
        int: number as int
    """
    assert type(numberStr) is str
    x = numberStr.replace(',', '')
    if 'K' in x:  # thousands
        if len(x) > 1:
            return int(float(x.replace('K', '')) * 1000)
        return 1000
    elif 'M' in x:  # millions
        if len(x) > 1:
            return int(float(x.replace('M', '')) * 1000000)
        return 1000000
    elif 'B' in x:  # billions
        return int(float(x.replace('B', '')) * 1000000000)
    else:
        return int(x)


def pageScraper(url: str, driver: webdriver) -> dict[str, None]:
    """
    Scrapes views, likes, comments, and game name from a url given a webdriver instance
    returns a dict with the keys 'likes', 'views', 'comments', 'game'

    Args:
        url (str): url for videopage
        driver (webdriver): selenium webdriver instance

    Returns:
        dict[str, None]: likes, views, comments, and name of game
    """
    driver.get(url)
    container = {
        'likes': None,
        'views': None,
        'comments': None,
        'game': None
    }
    substitutions = {'likes': '''video along with ([\d,KM]+)''',  # noqa:W605
                     'views': '''"views":\\{"simpleText":"\s*([\d,KM]+)\s*views"\\}''',  # noqa:W605
                     'comments': '''\\{"text":"Comments"\\}\\]\\},"contextualInfo":\\{"runs":\\[\\{"text":"([0-9,.KG]+)"\\}\\]\\}''',  # noqa:W605
                     'game': '''"simpleText":"([\w\\-.\\?\s:'"!\\$@&|\\+\\*]+)"\\},(?:"subtitle":\\{"simpleText":"[0-9]+"\\},)?"callToAction":\\{"runs":\\[\\{"text":"Browse game"\\}\\]\\}'''}  # noqa:W605
    for key in container:
        try:
            container[key] = re.findall(re.compile(
                substitutions[key]), driver.page_source)[0]
        except IndexError:
            if key == 'game':
                continue
            print(key, ' empty for ',
                  driver.title[:-10], ' (', url, ')', sep='')  # ! exception, continuing
            substitutions[key] = 0
            continue

    for key in container:
        if key == 'game':
            continue
        try:
            container[key] = formatNumber(container[key])
        except AssertionError:  # Nonetype, empty result
            pass
    return container


def videoListScraper(channelId: str, apiInstance) -> pd.DataFrame:
    """
    Given a channel id, returns a list of all videos uploaded by that channel

    Args:
        channelId (str): youtube channelID
        apiInstance (_type_): intialized youtube API instance

    Returns:
        pd.DataFrame: dataframe of videos on the channel
    """
    container = []
    # channel 'uploads' playlist is hardcoded as this
    uploadsId = 'UU' + channelId[2:]
    request = apiInstance.playlistItems().list(
        part='snippet', playlistId=uploadsId, maxResults=50)
    res = request.execute()
    for item in res['items']:
        data = {'channel': item['snippet']['channelTitle'], 'title': item['snippet']['title'],
                'videoID': item['snippet']['resourceId']['videoId'], 'date': item['snippet']['publishedAt']}
        container.append(data)
    currentToken = res['nextPageToken']
    # def generator():
    #     '''
    #     infinite generator for tqdm
    #     '''
    #     while True:
    #         yield
    # for _ in tqdm(generator()):
    t = tqdm(total=round(res['pageInfo']['totalResults']/50))
    while True:
        t.update(1)
        request = apiInstance.playlistItems().list(part='snippet', playlistId=uploadsId,
                                                   maxResults=50, pageToken=currentToken)
        # sleep(1)
        res = request.execute()
        for item in tqdm(res['items'], leave=False):
            data = {'channel': item['snippet']['channelTitle'], 'title': item['snippet']['title'],
                    'videoID': item['snippet']['resourceId']['videoId'], 'date': item['snippet']['publishedAt']}
            container.append(data)
        try:
            currentToken = res['nextPageToken']
        except:
            t.close()
            break
    df = pd.DataFrame(container)
    return df


def videoInfoScraper(dfWithVideoID: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of videos, returns likes, downloads, etc

    Args:
        dfWithVideoID (_type_): dataframe of videos # ! with  videoId at index 3

    Returns:
        pd.DataFrame: dataframe with scraped data
    """
    failed = False  # boolean for if it's incomplete
    try:
        dfWithVideoID = pd.read_pickle('./_videoInfoScrape.pkl')
    except:
        dfWithVideoID = dfWithVideoID.reindex(
            columns=dfWithVideoID.columns.tolist() + ['likes', 'comments', 'views', 'game'])
    try:
        driver.quit()
    except:
        pass
    driver = startDriver()
    t = tqdm(dfWithVideoID.itertuples(), total=dfWithVideoID.shape[0])
    for row in t:
        idx = row[0]
        # skipping rows for restart
        if (dfWithVideoID.loc[idx, 'likes'] > 0) or (dfWithVideoID.loc[idx, 'comments'] > 0) or (dfWithVideoID.loc[idx, 'views'] > 0):
            continue
        url = 'https://www.youtube.com/watch?v=' + row[3]
        try:
            result = func_timeout(90, pageScraper, args=(url, driver))
        except FunctionTimedOut:  # ! rate limiting catch
            print("Timeout @", idx)
            # saves at breakpoint
            dfWithVideoID.to_pickle('_videoInfoScrape.pkl')
            driver.quit()
            sleep(10)
            failed = True
            break
        for column in result:
            dfWithVideoID.at[idx, column] = result[column]
        t.set_postfix({"Page": driver.title[:-10]})
        # if idx + 1 % 500 == 0:
        #     # pp.pprint(result)
        #     driver.quit()
        #     sleep(2)
        #     driver = startDriver()
    if failed:
        return videoInfoScraper(dfWithVideoID)
    try:
        os.remove('_videoInfoScrape.pkl')  # backup
    except OSError:
        pass
    driver.quit()
    return dfWithVideoID


def streamerPipeline(channelID: str, apiInstance) -> pd.DataFrame:
    """
    Pipeline to get youtube videos for a streamer

    Args:
        channelId (str): youtube channelID
        apiInstance (_type_): intialized youtube API instance

    Raises:
        NameError: Can't find channel from ID

    Returns:
        pd.DataFrame: Dataframe with all videos and video details
    """
    try:
        videoDf = videoListScraper(channelID, apiInstance)
    except:
        print("Couldn't get videos")
        raise NameError
    print('Done getting videos for {}'.format(videoDf.channel.iloc[0]))
    df = videoInfoScraper(videoDf)
    return df
