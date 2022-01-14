import pandas as pd
from pandas.core import frame
from seaborn.palettes import dark_palette

from descriptive import Person_links



def get_fraction_df():

    df_n = pd.read_csv('Data/tab_n.csv')

    new_df = pd.DataFrame()

    category = []
    education_level = []
    value = []
    tabel = []
    fraction = []

    for name in ['werkschool', 'buren', 'familie', 'huishouden']:
        df = pd.read_csv(f"Data/tab_{name}.csv")    
        for geslacht, lft, etngrp in zip(df_n['geslacht'], df_n['lft'], df_n['etngrp']):
            print('\n')
            print(geslacht, lft, etngrp)
            
            for i in range(1,4):
                total = Person_links(df, geslacht,lft,None,etngrp)
                person_links = Person_links(df, geslacht,lft,i,etngrp)
                category.append((geslacht, lft, etngrp))
                education_level.append(i)
                value.append(sum(person_links.links['n']))
                tabel.append(name)

                if sum(total.links['n']):
                    fraction.append(sum(person_links.links['n'])/ sum(total.links['n']))
                else:
                    fraction.append(0)
                # print(sum(person_links.links['n']))


    new_df['Category'] = category
    new_df['Education_level'] = education_level
    new_df['n'] = value
    new_df['Table'] = tabel
    new_df['Fraction'] = fraction

    new_df.to_csv('n_per_group.csv')


def get_total_fraction():
    df = pd.read_csv('n_per_group.csv')
    df_n = pd.read_csv('Data/tab_n.csv')



    new_df = pd.DataFrame()
    category = []
    education_level = []
    value = []
    fraction = []
    for geslacht, lft, etngrp in zip(df_n['geslacht'], df_n['lft'], df_n['etngrp']):
        df_x = df[df['Category'] == f'{geslacht, lft, etngrp}']

        for i in range(1,4):     
            total = sum(df_x['Fraction'])

            s = sum(df_x[df_x['Education_level'] == i]['Fraction'])
            category.append((geslacht, lft, etngrp))
            education_level.append(i)

            if total:

                fraction.append(s/total)

            else:
                fraction.append(0)

            # print('\n')
    new_df['Category'] = category
    new_df['Education_level'] = education_level
    # new_df['n'] = value
    new_df['Fraction'] = fraction

    new_df.to_csv('n_per_group2.csv')


new_df = pd.DataFrame()
df = pd.read_csv('n_per_group2.csv')
df_n = pd.read_csv('Data/tab_n.csv')

geslacht = []
lft = []
etngrp = []
education = []
n = []
for geslacht_, lft_, etngrp_ in zip(df_n['geslacht'], df_n['lft'], df_n['etngrp']):
    for i in range(1,4): 
        education.append(i)
        geslacht.append(geslacht_)
        lft.append(lft_)
        etngrp.append(etngrp_)

        df_x = df_n[df_n['geslacht'] == geslacht_]
        df_x = df_x[df_x['lft'] == lft_]
        df_x = df_x[df_x['etngrp'] == etngrp_]

        n_ = df_x['n']

        df_f = df[df['Category'] == f'{geslacht_, lft_, etngrp_}']
        df_f = df_f[df_f['Education_level'] == i]

        fraction = df_f['Fraction']


        n.append(int(int(n_) * float(fraction)))

new_df['geslacht'] = geslacht
new_df['lft'] = lft
new_df['etngrp'] = etngrp
new_df['education'] = education   
new_df['n'] = n   
new_df.to_csv('tab_n.csv')