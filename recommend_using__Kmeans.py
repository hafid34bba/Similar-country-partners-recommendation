#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tkinter as tk
from tkinter import *


country_list = ['RépubliqueCentrafricaine', 'Qatar', 'Malawi',  'California',
                 'Haiti', 'AUSTRALIA', 'Brazil',  'Japan', 'Sénégal',
                'ShipVictoriaFallsZimbabwe',

                'Harare', 'Zimbabwe', 'UnitedKingdom', 'DemocraticRepublicofCongo',
                'BROOKLY NEW YORK UNITED STATE', 'China', 'BurkinaFaso', 'Israel', 'Sweden',
                 'Nairobi', 'Belarus', 'Turkey', 'India', 'New York',
                'The Netherlands', 'Switzerland', 'Australia', 'Nairobi-Kenya', 'Geneva-SWITZERLAND',
                'Mongolia', 'Mexico', 'Spain', 'UAE', 'Austria', 'USA', 'RépubliquedeGuinée', 'France',
                'Philippines', 'CAUSA', 'TheNetherlands', 'theNetherlands', 'Kenya', 'RussianFederation',
                'Egypt', 'London', 'Geneva', 'Thailand', 'UnitedStates', 'Johannesburg', 'SaudiArabia',
                'Colombia', 'Belgium', 'morocco',  'UK', 'cabo verde',
                'UnitedArabEmirates',   'Finland', 'Italy',
                'Morocco', 'UGANDA', 'Bamako', 'Abuja', 'Germany', 'NEW YORK', 'Vietnam',
                'Permanent Mission of Liechtenstein to the United Nations in New York', 'Nigeria', 'Ecuador',
                '61987', 'Paris', 'Netherlands', 'theUnitedKingdon', 'Perú', 'Guatemala', 'DRCongo', 'Chad',
                'Cyprus', 'Spain(IAHRSecretariat)', 'Malaysia', 'SWEDEN', 'SouthAfrica', 'NIGERIA',
                'Milan - Italy', 'Korea', 'Kampala Uganda', 'Lebanon', 'Poland', 'Sweden)', 'TURKEY', 'Peru',
                'Mexico City', 'Dhaka | Bangladesh', 'THE GAMBIA', 'Lima', 'Mali', 'México',
                'LAPFUND GARDENS- NAIROBI KENYA', 'Zvishavane', 'Uganda', 'Libreville-Gabon',
                'Mauritius', 'KAMPALA - UGANDA', 'CostaRica', 'HazinaTowers', 'Afghanistan', 'NIGERIA.',
                'NEW DELHI', 'Honduras', 'MEx', 'Denmark', 'Lomé-Togo', 'MD', 'botswana',
                'UNDP Mexico Country Office', 'GREECE', 'City of Buenos Aires', 'Justice Canada',
                'St. John’s Antigua', 'SriLanka', 'Ankara Turkey', 'lisbon', 'UnitedStatesofAmerica',
                'Argentina.', 'Washington DC', 'The Hague', 'Liberia', 'U.K.', 'Bogura', 'Chile',
                'LEBANON)', 'RJ-Brazil', 'Canada', 'Turkiye', 'RépubliqueDémocratiqueduCongo']


def work():
    country = variable.get()

    descruption_input = desc.get()

    print(country,descruption_input)

    # # Read and clean data

    # In[15]:

    data = pd.read_csv('Cleaned_SDG_Partnerships.csv')
    data.head()

    # In[16]:

    data = data.drop_duplicates(subset='description', keep="last")

    # In[17]:

    data[data['contact_information'].notna()]['contact_information'].unique()

    # In[18]:

    data = data[data['contact_information'].notna()]
    data['country_name'] = data['contact_information'].apply(
        lambda x: x.replace('/', ',').split(',')[-1].replace(' ', '') if len(x.replace('/', ',').split(',')) > 1 else x)
    data.head()

    # # Get needed data according to country and get description ...

    # In[19]:

    needed_data = data[['country_name', 'description', 'contact_information', "link", 'name_of_partnership']]
    needed_data['description'] = needed_data['description'].apply(lambda x: ' '.join(x.split()[1:]))

    # In[20]:

    needed_data = needed_data[needed_data['country_name'] == country]

    # # Tokenize data

    # In[22]:

    import nltk
    import re

    tokenized_data = []

    for index, row in needed_data.iterrows():
        tokenized_data.append(nltk.word_tokenize(
            re.sub('[^A-Za-z0-9|-|é]+', '', row['description'].replace(' ', 'é')).replace('é', ' ').lower()))

    descruption_input = nltk.word_tokenize(
        re.sub('[^A-Za-z0-9|-|é]+', '', descruption_input.replace(' ', 'é')).replace('é', ' ').lower())

    # In[23]:

    needed_data['toknized_desc'] = tokenized_data

    # # Remove stop word

    # In[24]:

    import nltk
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))

    filterd_sentences = []

    for sent in tokenized_data:
        flt = []
        for w in sent:
            if w not in stop_words:
                flt.append(w)

        filterd_sentences.append(flt)

    filterd_des = []
    for word in descruption_input:
        if word not in stop_words:
            filterd_des.append(word)

    # # Apply lemmatization

    # In[25]:

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    lim_result = []

    for sent in filterd_sentences:
        tmp = []
        for word in sent:
            tmp.append(lemmatizer.lemmatize(word))
        lim_result.append(tmp)

    lim_input_des = []

    for word in filterd_des:
        lim_input_des.append(lemmatizer.lemmatize(word))

    # # BOW

    # In[26]:

    Bow_data = []
    all_words = []

    for sent in lim_result:
        tmp = {}
        for word in sent:
            if word not in tmp:
                tmp[word] = 0
            tmp[word] += 1
            if word not in all_words:
                all_words.append(word)
        Bow_data.append(tmp)

    bow_input_desc = {}

    for word in lim_input_des:
        if word in all_words:
            if word not in bow_input_desc:
                bow_input_desc[word] = 0

            bow_input_desc[word] += 1

    # # Get term frequency

    # In[27]:

    TF_data = []

    for sent in Bow_data:
        tmp = []
        for word in all_words:
            if word in sent:
                tmp.append(sent[word])
            else:
                tmp.append(0)
        TF_data.append(tmp)

    TF_input_desc = []

    for word in all_words:
        if word in bow_input_desc:
            TF_input_desc.append(bow_input_desc[word])
        else:
            TF_input_desc.append(0)

    # In[28]:

    len(TF_data[0])

    # # Train kmeans clustering

    # In[29]:

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(TF_data)

    # In[30]:

    prd = kmeans.predict([TF_input_desc])
    prd

    # # get nearst partners and get the score

    # In[31]:

    neighs_ind = [j for j in range(len(kmeans.labels_)) if kmeans.labels_[j] == prd]

    # In[32]:

    neighs = []

    for idx in neighs_ind:
        neighs.append(TF_data[idx])

    # In[33]:

    name_of_partnership = list(needed_data['name_of_partnership'])
    contact_information = list(needed_data['contact_information'])
    link = list(needed_data['link'])

    # In[34]:

    from scipy.stats import pearsonr

    neigh_scores = {}

    for i in range(len(neighs)):
        neigh_scores[neighs_ind[i]] = pearsonr(TF_data[0], neighs[i])[0] * 100

    # In[35]:

    neigh_scores = {k: v for k, v in sorted(neigh_scores.items(), key=lambda item: item[1], reverse=True)}

    # # print results

    # In[36]:

    most_nearst_partners = list(neigh_scores.keys())
    scores = list(neigh_scores.values())


    sh = []
    for j in range(10):
        if j >= len(most_nearst_partners):
            break
        id_ = most_nearst_partners[j]
        sh.append('Name : '+str( name_of_partnership[id_]))
        sh.append('Contact_information : ' +contact_information[id_])
        sh.append('Link : '+link[id_] )
        sh.append('Score : ' + str(abs(scores[j])))
        sh.append('\n ------ ------- \n')

    scrollbar = Scrollbar()
    for s in sh:
        T.insert(END, s)


    window.mainloop()






window = tk.Tk()
window.title('Results')
window.geometry("460x320+130+180")


labelText=StringVar()
labelText.set("Enter Description")
labelDir=Label(window, textvariable=labelText, height=4)
labelDir.grid(row=1,column=0)

desc=StringVar(None)
desc=Entry(window,textvariable=desc,width=20)
desc.grid(row=1,column=2)

country_l=StringVar()
country_l.set("Select country")
country_l=Label(window, textvariable=country_l, height=4)
country_l.grid(row=3,column=0)

variable = StringVar(window)
variable.set("France")

country = OptionMenu(window, variable, *country_list)
country.grid(row=3,column=2)
# Insert The Fact.
MyButton1 = Button(window, text="Submit", width=10, command=work)
MyButton1.grid(row=5,column=1)

labelText=StringVar()
labelText.set("Here partners with similar goals")
labelDir=Label(window, textvariable=labelText, height=4)
labelDir.grid(row=6,column=0)

T = Listbox(window, width=60, height=6)
T.grid(row=8,column=0,columnspan = 4)
# Create label
l = Label(window, text="Here is partners with similar goals")

yscroll = tk.Scrollbar(command=T.yview, orient=tk.VERTICAL)
yscroll.grid(row=8, column=4, sticky='ns')
T.configure(yscrollcommand=yscroll.set)




lines = 100
T.yview_scroll(lines, 'units')




window.mainloop()