# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:37:54 2020

@author: jashi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tit = sns.load_dataset('titanic')

# %% Preprocesing data

tit.info()

tit = tit[tit['embark_town'].notnull()]


# %%

temp_males = tit[tit['who']=='man']
temp_females = tit[tit['who']=='woman']

alone_in_the_city_males = temp_males.pivot_table(values='survived',
         index='embark_town', columns='alone', aggfunc= 'mean')
alone_in_the_city_females = temp_females.pivot_table(values='survived',
         index='embark_town', columns='alone', aggfunc= 'mean')

city_class = tit.pivot_table(columns='embark_town', index='class',
                             values='survived', aggfunc='count')

city_class.plot(kind='bar')
alone_in_the_city_males.plot(kind='bar', 
            title='How being alone affects chances to survive (men)')
alone_in_the_city_females.plot(kind='bar',
            title='How being alone affects chances to survive (women)')

# %%

temp_males = tit[tit['who']=='man']
temp_females = tit[tit['who']=='woman']

alone_in_the_city_males = temp_males.pivot_table(values='survived',
         index='class', columns='alone', aggfunc= 'mean')
alone_in_the_city_females = temp_females.pivot_table(values='survived',
         index='class', columns='alone', aggfunc= 'mean')

alone_in_the_city_males.plot(kind='bar', 
            title='How being alone affects chances to survive (men)')
alone_in_the_city_females.plot(kind='bar',
            title='How being alone affects chances to survive (women)')

# %%

fare = tit['fare']
tit['qbin'] = pd.qcut(fare, 5)
pay_to_survive = tit.pivot_table(index = 'qbin',
                      columns='sex', values = 'survived', aggfunc='mean')
pay_to_survive.plot()

# %%

survived = tit[tit['survived']==1]

# %%
sns.set()
sns.relplot(data=tit, x='age', y='fare', palette='viridis', 
            hue=tit.sex.tolist())
