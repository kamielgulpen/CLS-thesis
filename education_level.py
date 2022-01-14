import pandas as pd
from pandas.core import frame
from seaborn.palettes import dark_palette

from descriptive import Person_links



def get_fraction_df():
    '''
    Returns the fraction that each education level is represented in each group

    Example:

    0,"('Vrouw', '[20,30)', 'Overig')",1,63710,werkschool,0.048241761570147805
    1,"('Vrouw', '[20,30)', 'Overig')",2,657160,werkschool,0.4976072207414587
    2,"('Vrouw', '[20,30)', 'Overig')",3,599770,werkschool,0.4541510176883935

    Here in the group ('Vrouw', '[20,30)', 'Overig') is education level of 1 represented by 
    a fraction of 0.048, a education level of 2 represented by 0.497 and a education level of 3 by 0.455
    '''
    # Import datafile n file
    df_n = pd.read_csv('Data/tab_n.csv')

    new_df = pd.DataFrame()

    # Initialize lists
    category = []
    education_level = []
    value = []
    tabel = []
    fraction = []
    
    # Loops through all layers
    for name in ['werkschool', 'buren', 'familie', 'huishouden']:
        df = pd.read_csv(f"Data/tab_{name}.csv")    

        # Loops through all groups
        for geslacht, lft, etngrp in zip(df_n['geslacht'], df_n['lft'], df_n['etngrp']):
            
            # Looks at the fraction that a person education level is represented in the total group
            for i in range(1,4):

                # Get the total group
                total = Person_links(df, geslacht,lft,None,etngrp)

                # Gets the persons of the group with education level i
                person_links = Person_links(df, geslacht,lft,i,etngrp)

                # Appends category, education level, amount of people with i education level and layer
                category.append((geslacht, lft, etngrp))
                education_level.append(i)
                value.append(sum(person_links.links['n']))
                tabel.append(name)

                # Divides amount of people with i education level by total people of the group
                # Appends fraction
                if sum(total.links['n']):
                    fraction.append(sum(person_links.links['n'])/ sum(total.links['n']))
                else:
                    fraction.append(0)



    new_df['Category'] = category
    new_df['Education_level'] = education_level
    new_df['n'] = value
    new_df['Table'] = tabel
    new_df['Fraction'] = fraction

    new_df.to_csv('n_per_group.csv')


def get_total_fraction():
    '''
    returns the average of fractions over the 4 different layers obtained in def get_fraction_df()

    Example: 
    0,"('Vrouw', '[20,30)', 'Overig')",1,0.11763628546214247
    1,"('Vrouw', '[20,30)', 'Overig')",2,0.4781271443504522
    2,"('Vrouw', '[20,30)', 'Overig')",3,0.40423657018740533

    This shows the average education fractions based on 4 different layers of the group the 'Vrouw', '[20,30)', 'Overig'
    '''

    # Import dataframes
    df = pd.read_csv('n_per_group.csv')
    df_n = pd.read_csv('Data/tab_n.csv')


    # Initializes lists and dataframe
    new_df = pd.DataFrame()
    category = []
    education_level = []
    value = []
    fraction = []
    
    # Loops through all groups
    for geslacht, lft, etngrp in zip(df_n['geslacht'], df_n['lft'], df_n['etngrp']):

        # make dataframe of the values which are equal to the ith group of the iteration
        df_x = df[df['Category'] == f'{geslacht, lft, etngrp}']

        # Loop over the education levels
        for i in range(1,4):     

            # Get sum value of fractions, which is always 4 (4 layers)
            total = sum(df_x['Fraction'])

            # Takes the summation of education level i over the four layers
            s = sum(df_x[df_x['Education_level'] == i]['Fraction'])

            category.append((geslacht, lft, etngrp))
            education_level.append(i)

            # Divide the summation by the total to get a average fraction
            if total:
                fraction.append(s/total)

            else:
                fraction.append(0)


    new_df['Category'] = category
    new_df['Education_level'] = education_level
    # new_df['n'] = value
    new_df['Fraction'] = fraction

    new_df.to_csv('n_per_group2.csv')



def finish()
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