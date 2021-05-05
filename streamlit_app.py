#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Gapminder Dashboard')
st.markdown('### Bubble chart depicting relationship between GNI per capita and Life expectancy')


@st.cache
def load_and_preprocess_data(data1, data2, data3):
    """Loads and preprocess the data from 3 different (ordered!) sources and then merges them"""
    data1 = pd.read_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/{}'.format(data1)).set_index('country') #reading the data
    data2 = pd.read_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/{}'.format(data2)).set_index('country')
    data3 = pd.read_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/{}'.format(data3)).set_index('country')
    data1.bfill(axis=1, inplace=True) #backward and then forward filling for each datasource
    data1.ffill(axis=1, inplace=True)
    data2.bfill(axis=1, inplace=True)
    data2.ffill(axis=1, inplace=True)
    data3.bfill(axis=1, inplace=True)
    data3.ffill(axis=1, inplace=True)
    data1['country'] = data1.index #creating a new column with country names
    data2['country'] = data2.index
    data3['country'] = data3.index
    data1.reset_index(drop=True, inplace=True) #resetting the index
    data2.reset_index(drop=True, inplace=True)
    data3.reset_index(drop=True, inplace=True)
    data1 = data1.melt(id_vars='country', var_name='year', value_name='gni_per_capita') #redesigning the dataframe so that rows with column names represent one instance in a row
    data2 = data2.melt(id_vars='country', var_name='year', value_name='life_exp')
    data3 = data3.melt(id_vars='country', var_name='year', value_name='population')
    dfs = data1.merge(data2, on=['country','year']).merge(data3, on=['country','year']) #merging dataframes
    dfs['gni_per_capita'] = np.log(dfs['gni_per_capita']) #converting gni to logarithmic scale
    return dfs

dfs = load_and_preprocess_data('gnipercapita_ppp.csv', 'life_expectancy_years.csv', 'population_total.csv')

year = st.sidebar.slider(min_value=1990, max_value=2020, step=1, label='Year')
country_selection = list(np.unique(dfs.country))
countries = st.multiselect(label = 'Choose one or more countries', options=country_selection)

query = dfs[(dfs['year'] == str(year))  & (dfs['country'].isin(countries))]

plt.figure(figsize=(8,6))
sns.scatterplot(x='gni_per_capita', y='life_exp', size = 'population', hue='country', data=query, legend='full')
plt.xlim(0, dfs.gni_per_capita.max())
plt.ylim(0, dfs.life_exp.max())
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(xlabel='Logarithmic GNI per capita (PPP adjusted)',fontsize=16)
plt.ylabel(ylabel='Life expectancy',fontsize=16)
plt.legend(prop={'size':12})
plt.show()
st.pyplot()