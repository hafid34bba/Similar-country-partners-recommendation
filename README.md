# Similar-country-partners-recommendation

This project aim to recommend similair parnters for user according to input description and selected <br>
country name using NLP technics and clustering algorithms.<br>

data used for that project is Scopus List of Indexed Journals

# What Are Scopus Journals?
Scopus is one of the largest, most reputable abstract and citation databases for academic literature. It contains over 40,000 titles from more than 10,000 international publishers, and nearly 35,000 of these publications are peer-reviewed. Scopus covers various formats (books, journals, conference papers, etc.) in the fields of science, technology, medicine, social sciences, and arts and humanities.

# Steps involved while building the model:
1-	Get the country name and description from the user
2-	Read and clean the data <br>
- read data<br>
- remove all rows with Nan description<br>
- extract country name from contact_information <br>
- select rows where country_name = input_country_name<br>
       
3-	Apply tokenization<br>
4-	Remove stop words<br>
5-	Apply lemmatization <br>
6-	Get Bow<br>
7-	Display wordcloud<br>
8-	Get term frequency using Bow<br>
9-	Train clustering model (kmeans / hierarchical clustering)<br>
10-	Get the nearest partners and scores <br>
11-	Display nearst partners <br>


# GUI:
<img src="https://github.com/hafid34bba/Similar-country-partners-recommendation/blob/main/design.png">





