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

read_and_cache_csv = st.cache(pd.read_csv)

gni_data = read_and_cache_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/gnipercapita_ppp.csv').set_index('country')
life_exp_data = read_and_cache_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/life_expectancy_years.csv').set_index('country')
pop_data = read_and_cache_csv('/Users/vahe_shelunts/Desktop/HWR Winter 2020:2021/Enterprise architectures/gapminder_dashboard/app/population_total.csv').set_index('country')



#backward filling and then forward filling on column values
gni_data.bfill(axis=1, inplace=True) 
gni_data.head()


gni_data.ffill(axis=1, inplace=True)
gni_data.info()


life_exp_data.bfill(axis=1, inplace=True)
life_exp_data.ffill(axis=1, inplace=True)
life_exp_data.info()


pop_data.bfill(axis=1, inplace=True)
pop_data.ffill(axis=1, inplace=True)
pop_data.info()


#creating a new column with country names
gni_data['country'] = gni_data.index
life_exp_data['country'] = life_exp_data.index
pop_data['country'] = pop_data.index


#resetting the index
gni_data.reset_index(drop=True, inplace=True)
life_exp_data.reset_index(drop=True, inplace=True)
pop_data.reset_index(drop=True, inplace=True)


gni_data = gni_data.melt(id_vars='country', var_name='year', value_name='gni_per_capita')
life_exp_data = life_exp_data.melt(id_vars='country', var_name='year', value_name='life_exp')
pop_data = pop_data.melt(id_vars='country', var_name='year', value_name='population')


pop_data.info()
dfs = gni_data.merge(life_exp_data, on=['country','year']).merge(pop_data, on=['country','year'])


dfs['gni_per_capita'] = np.log(dfs['gni_per_capita'])


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