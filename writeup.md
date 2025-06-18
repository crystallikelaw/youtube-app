# Youtube for Content Creators

#### Kapilan Mahalingam, January 2023

---
I set out to develop a web app for youtube content creators (gamers, specifically) to look at for advice on what games to play to incrase viewership. I acquired relevant data, set up an analytics dashboard and provided a recommendation system for games that might be productive to make videos for.

### Design

As I'm making something for content creators, direct data from youtube was the most important. Consequently I scraped to overcome API limitations. I also focused on both game characteristics as well. For compuational and deployment reasons, I used matrix factorization methods and pretrained models to limit filesize and memory footprint. Due to the vagaries of web scraping I put in extensive error handling in the data aquisition to future proof it (and still work if API's get depreciated).

### Data

Data from youtube API (list of channel videos) and scraping youtube for the rest. Steam ap ID's came for two three different implementations with a scraper as fallback. I used both SQL databases and serialization to store data (the latter in an attempt to optimize memory) for deployment.

### Algorithms

I made a neural network which I gave up on at deployment. I used matrix based recommendation systems after signifcant data preprocessing, using NLTK and surprise for their implementations.

### Tools and Communication

I used streamlit to develop a web app [here](https://crystallikelaw-streamlit6-try1-r4qgjt.streamlit.app/). The visualizations were done in matplotlib with some seaborn thrown in.
