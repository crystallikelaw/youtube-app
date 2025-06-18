import pickle

import altair as alt
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import correlation, cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

import streamlit as st

app_mode = st.sidebar.radio('Select Page', ['Dashboard', 'Recommendation'])


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

df_vids = pd.read_pickle('./dfvids.p')

if app_mode == 'Dashboard':
    # topfilter = st.sidebar.selectbox("", ["Youtubers", "Games"])
    filter = st.sidebar.radio("Scope", ["Full Dataset", "Single Channel"])
    if filter == 'Full Dataset':

        st.title('Dashboard (Full Dataset):')
        st.markdown("## Total Statistics")
        st.write("Here are some statists for the entire dataset")
        allvids = st.selectbox(
            "", ["Views", "Engagement", "Output", "Game Variety"])
        # , "Genres", "Games"
        if allvids == 'Views':
            st.markdown('#### Most Viewed Channels :')
            st.write('These are channels that have the most views, in total.')
            tviews = df_vids.groupby(by=['Channel Title']).Views.sum(
            ).sort_values(ascending=False).reset_index()
            # st.write(tviews)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews, x='Channel Title',
                             y='Views', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'B' for x in ax.get_yticks() / 1000000000]
            ax.set(ylabel="Total Views (in billions)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)

            st.markdown('#### Most Views per video :')
            st.write("As a contrast, these are the channels with the most views per video. Channels who churn out lots of low popularity videos will suffer in these rankings.")
            tviews2 = df_vids.groupby(by=['Channel Title']).Views.sum(
            ) / df_vids.groupby(by=['Channel Title']).Views.count()
            tviews2 = tviews2.reset_index().sort_values(
                by=['Views'], ascending=False)
            # st.write(tviews2)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews2, x='Channel Title',
                             y='Views', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set(ylabel="Average Views (in millions)")
            # ax.set_yticklabels(xlabels)
            st.pyplot(plt)

        if allvids == 'Engagement':
            st.write("Engagement, defined as (likes+comments), is an important factor both in terms of ad revenue generated per view, as well as how high up in recommendations/searches the video appears.")
            st.markdown('#### Channels with the most total engagements :')
            st.write('Again, these are channels with')
            tviews = df_vids.groupby(by=['Channel Title']).Engagement.sum(
            ).sort_values(ascending=False).reset_index()
            # st.write(tviews)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews, x='Channel Title',
                             y='Engagement', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'K' for x in ax.get_yticks() / 1000]
            ax.set(ylabel="Total Engagement (in billions)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)

            st.markdown('#### Most Engagement per video :')
            st.write("Again, a measure of the 'quality' of individual videos.")
            tviews2 = df_vids.groupby(by=['Channel Title']).Engagement.sum(
            ) / df_vids.groupby(by=['Channel Title']).Engagement.count()

            tviews2 = tviews2.reset_index().sort_values(
                by=['Engagement'], ascending=False)
            # st.write(tviews2)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews2, x='Channel Title',
                             y='Engagement', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'K' for x in ax.get_yticks() / 1000]
            ax.set(ylabel="Average Engagement (in millions)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)

            st.markdown('#### Most Engagement per view :')
            st.write("This is the percentage of views that result in an engagement. In addition to engagement, this is a measure of invested your viewership is; higher investment correlates with lower variance over time (in views, comments, etc.)")
            tviews2 = df_vids.groupby(by=['Channel Title']).Engagement.sum(
            ) / df_vids.groupby(by=['Channel Title']).Views.sum()
            tviews2 = tviews2.reset_index()
            tviews2.columns = ['Channel Title', 'Engagement']
            tviews2 = tviews2.sort_values(by=['Engagement'], ascending=False)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews2, x='Channel Title',
                             y='Engagement', color='steelblue')
            xlabels = ['{:,.0f}%'.format(x) for x in ax.get_yticks() * 100]
            ax.set(ylabel="Engagement per view (percent)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)
        if allvids == 'Output':
            st.write("This is how prolific each channel is.")
            st.markdown('#### Channels with the most videos :')
            tviews = df_vids.groupby(by=['Channel Title']).Engagement.count(
            ).sort_values(ascending=False).reset_index()
            # st.write(tviews)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews, x='Channel Title',
                             y='Engagement', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'K' for x in ax.get_yticks() / 1000]
            ax.set(ylabel="Videos produced till date (in billions)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)
        if allvids == 'Game Variety':
            st.write("This is how many different games a channel features; are they a generalist playing lots of different games, or a specialist, focusing on a few games they like?")
            st.markdown('#### Channels with the most unique games :')
            tviews = tviews = df_vids.groupby(by=['Channel Title'])[
                "Game Played"].nunique().sort_values(ascending=False).reset_index()
            # st.write(tviews)
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.barplot(tviews, x='Channel Title',
                             y='Game Played', color='steelblue')
            # xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
            ax.set(ylabel="Different Games Played")
            # ax.set_yticklabels(xlabels)
            st.pyplot(plt)
        if allvids == 'Genres':
            pass

    if filter == 'Single Channel':
        ytber = st.sidebar.selectbox("Youtuber", ('AngryJoeShow', 'H2ODelirious', 'Markiplier', 'Mathas',
                                                  'Many A True Nerd', 'theRadBrad', 'Smosh Games',
                                                  'Splattercat', 'DanTDM', 'quill18', 'jacksepticeye',
                                                  'SeaNanners', 'PartyElite', 'VanossGaming', 'Blitz',
                                                  'KatherineOfSky', 'SmallAnt'))

        st.title('Dashboard for {}:'.format(ytber))
        st.write("Here are some more in depth statistics for {}".format(ytber))
        allvids = st.selectbox(
            "", ["Best Videos", "Distributions", "Over Time", "Best Games", "Correlation"])
        ydf = df_vids[df_vids['Channel Title'] == ytber]

        if allvids == 'Best Videos':
            st.write(ydf[['Video Title', 'Likes', 'Comments', 'Views', 'Game Played',
                     'Engagement']].sort_values(by=['Views'], ascending=False).head(20))
            # st.write(ydf.columns)

        if allvids == 'Distributions':
            var = st.radio('Select Variable', [
                           'Views', 'Likes', 'Comments', 'Engagement'])
            # st.write(ydf)
            fig = plt.figure(figsize=(10, 5))
            ax = sns.displot(data=ydf, x=var, color='steelblue', kind='kde')
            ax.set(xlabel=var)
            st.pyplot(plt)

        if allvids == 'Over Time':
            st.markdown('### Views over time')
            # st.write(y_df.head())
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            # ax = sns.scatterplot(ydf, x='Upload Date', y='Views', color='steelblue')
            ax = sns.regplot(ydf, x='Upload Year', y='Views',
                             color='steelblue', scatter=True)
            ylabels = ['{:,.0f}'.format(
                x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set(ylabel="Average Views (in millions)")
            ax.set_yticklabels(ylabels)
            st.pyplot(plt)

            st.markdown('### Engagement over time')
            # st.write(y_df.head())
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=40, fontsize=8)
            ax = sns.regplot(ydf, x='Upload Year', y='Engagement',
                             color='steelblue', scatter=True)
            # ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1000000]
            ax.set(ylabel="Average Engagement")
            # ax.set_yticklabels(ylabels)
            st.pyplot(plt)
        if allvids == 'Best Games':
            tviews = pd.DataFrame(ydf.groupby(by=['Game Played']).Views.sum(
            ) / ydf.groupby(by=['Game Played']).Views.count())
            tviews['Engagement'] = ydf.groupby(by=['Game Played']).Engagement.sum(
            ) / ydf.groupby(by=['Game Played']).Views.count()
            tviews['Likes'] = ydf.groupby(by=['Game Played']).Likes.sum(
            ) / ydf.groupby(by=['Game Played']).Views.count()
            tviews['Comments'] = ydf.groupby(by=['Game Played']).Comments.sum(
            ) / ydf.groupby(by=['Game Played']).Views.count()
            tviews = tviews.sort_values(
                by=['Views'], ascending=False).reset_index()
            tviews.rename(columns={'Views': "Average Views", "Likes": "Average Likes",
                          "Comments": "Average Comments", "Engagement": "Average Engagement"}, inplace=True)
            st.write(tviews.head(11))
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=60, fontsize=8)
            ax = sns.barplot(tviews.head(20), x='Game Played',
                             y='Average Views', color='steelblue')
            xlabels = ['{:,.0f}'.format(
                x) + 'K' for x in ax.get_yticks() / 1000]
            ax.set(ylabel="Average Views (in thousands)")
            ax.set_yticklabels(xlabels)
            st.pyplot(plt)
            tviews = tviews.sort_values(
                by=['Average Engagement'], ascending=False).reset_index()
            fig = plt.figure(figsize=(10, 5))
            plt.xticks(rotation=60, fontsize=8)
            ax = sns.barplot(tviews.head(20), x='Game Played',
                             y='Average Engagement', color='steelblue')
            # xlabels = ['{:,.0f}'.format(x) + 'B' for x in ax.get_yticks()/1000000000]
            ax.set(ylabel="Average Engagement")
            # ax.set_yticklabels(xlabels)
            st.pyplot(plt)
        if allvids == 'Correlation':
            # x = 'Views'
            # y = 'Likes'
            st.write(ydf.corr())


if app_mode == 'Recommendation':
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    df_games = pd.read_pickle('./dfgames.p')
    st.write(df_games.head())

    @st.cache
    def nltkTokenize(text: str) -> str:
        """
        Returns tokenized, lemmatized workds with a few common abriviations

        Args:
            text (str): input text

        Returns:
            str: string of tokens
        """
        temp_sent = []
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)
        VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        for i, word in enumerate(words):
            if tags[i][1] in VERB_CODES:
                lemmatized = lemmatizer.lemmatize(word, 'v')
            else:
                lemmatized = lemmatizer.lemmatize(word)
            if lemmatized not in stop_words and lemmatized.isalpha():
                temp_sent.append(lemmatized)

        finalsent = ' '.join(temp_sent)
        finalsent = finalsent.replace("n't", " not")
        finalsent = finalsent.replace("'m", " am")
        finalsent = finalsent.replace("'re", " are")
        finalsent = finalsent.replace("'ll", " will")
        finalsent = finalsent.replace("'ve", " have")
        finalsent = finalsent.replace("'d", " would")
        return finalsent

    desc = df_games.Description.apply(nltkTokenize)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(desc)
    cos_similar = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_games.index, index=df_games['Game Title'])

    def getRecs(titlelist: list[str], count: int = 10, yt=0) -> list:
        """
        Returns recommendations based on cosine similarity

        Args:
            titlelist (list[str]): List of Game titles

        Returns:
            list: List of Recommendations
        """
        assert count < cos_similar.shape[0] - len(titlelist)
        idxs = []
        if yt:
            titlelist = df_vids[df_vids['Channel Title'] == yt].groupby(
                by=['Game Played']).Views.sum().sort_values(ascending=False).head(10).index
        for title in titlelist:
            try:
                idxs.append(indices[title])
            except KeyError:  # !
                print('"', title, '" not found, ignoring', sep='')
                continue
        if not idxs:
            print('No valid input, exiting')
            return 0
        similarity_scores = pd.Series(cos_similar[idxs[0]])
        for idx in idxs[1:]:
            similarity_scores += pd.Series(cos_similar[idx])
        if yt:
            newidxs = []
            for title in df_vids[df_vids['Channel Title'] == yt]:
                try:
                    newidxs.append(indices[title])
                except KeyError:  # !
                    continue
            idxs = idxs + newidxs
        similarity_scores.drop(labels=idxs, axis=0, inplace=True)
        similarity_scores.sort_values(ascending=False, inplace=True)
        topRecs = list(similarity_scores.iloc[:count].index)
        return [(df_games['Game Title'].loc[rec], df_games['header_image'].loc[rec]) for rec in topRecs]

    st.markdown("## Content based natural language recommender")
    st.write("This is a basic content-based recommender system to recommend similar games. This analyses games using their titles and descriptions.")

    options = st.multiselect('Enter some games!', df_games['Game Title'])
    if not options:
        tube = st.selectbox("Or pick a youtuber from the list", ('AngryJoeShow', 'H2ODelirious', 'Markiplier', 'Mathas',
                                                                 'Many A True Nerd', 'theRadBrad', 'Smosh Games',
                                                                 'Splattercat', 'DanTDM', 'quill18', 'jacksepticeye',
                                                                 'SeaNanners', 'PartyElite', 'VanossGaming', 'Blitz',
                                                                 'KatherineOfSky', 'SmallAnt'))

    st.write()
    count = st.slider('Number of recommendations?', 1, 100, 1)
    if st.button('Get {} reccomendations'.format(count)):
        if options:
            t = getRecs(options, count=count)
            for i in range(len(t)):
                st.image(t[i][1], width=200)
                st.write(t[i][0])
        else:
            t = getRecs([], count=count, yt=tube)
            for i in range(len(t)):
                st.image(t[i][1], width=200)
                st.write(t[i][0])
