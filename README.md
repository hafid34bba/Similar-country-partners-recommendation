# Similar-country-partners-recommendation

This project aim to recommend similair parnters for user according to input description and selected 
country name using NLP technics and clustering algorithms.

# Steps involved while building the model:
1-	Get the country name and description from the user
2-	Read and clean the data 
    -read data
    - remove all rows with Nan description
    - extract country name from contact_information 
    - select rows where country_name = input_country_name
       
3-	Apply tokenization
4-	Remove stop words
5-	Apply lemmatization 
6-	Get Bow
7-	Display wordcloud
8-	Get term frequency using Bow
9-	Train clustering model (kmeans / hierarchical clustering)
10-	Get the nearest partners and scores 
11-	Display nearst partners 




