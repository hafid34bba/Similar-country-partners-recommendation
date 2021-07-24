# Similar-country-partners-recommendation

This project aim to recommend similair parnters for user according to input description and selected 
country name using NLP technics and clustering algorithms.

# Steps involved while building the model:
1-	Get the country name and description from the user
2-	Read and clean the data <br>
    &nbsp;&nbsp;-read data<br>
    &nbsp&nbsp- remove all rows with Nan description<br>
    &nbsp&nbsp- extract country name from contact_information <br>
    &nbsp&nbsp- select rows where country_name = input_country_name<br>
       
3-	Apply tokenization<br>
4-	Remove stop words<br>
5-	Apply lemmatization <br>
6-	Get Bow<br>
7-	Display wordcloud<br>
8-	Get term frequency using Bow<br>
9-	Train clustering model (kmeans / hierarchical clustering)<br>
10-	Get the nearest partners and scores <br>
11-	Display nearst partners <br>





