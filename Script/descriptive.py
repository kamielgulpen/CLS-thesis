from importlib.metadata import distribution
from unicodedata import category
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.font_manager as fm#  Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.font"] = "Computer Modern Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# from inequalipy import *

def get_total_edges(name):
    df = pd.read_csv(f'Data/tab_{name}.csv')
    return sum(df['n'])

def get_total_nodes():
    df = pd.read_csv('Data/tab_n_(with oplniv).csv')
    return sum(df['n'])

def get_groups():
    df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')

    gender = df_n['geslacht'].unique()
    age = df_n['lft'].unique()
    etn = df_n['etngrp'].unique()
    edu = df_n['oplniv'].unique()

    return gender, age, etn, edu

def count_values(df, name):
    '''
    Counting the values in the dataframe and export it to a csv file
    '''
    concated_df = pd.DataFrame()

    for column in df.columns:
        if column == 'n':
            break
        
        concated_df = pd.concat([concated_df, df[column].value_counts()], axis=1)


    concated_df.to_csv(f'Data/Descriptives/tab_{name}/counts.csv')

def mean_columns(df, name):
    '''
    Calculates the means for each column in the dataframe and exports
    them to csv files
    '''
    for column in df.columns:
        mean_column = df.groupby(column).mean()

        mean_column.to_csv(f'Data/Descriptives/tab_{name}/mean_{column}.csv')
        print(mean_column)


class Person_links:
    '''
    Makes a class for group of persons
    '''
    def __init__(self, df, gender, age, education, ethnicity):
        self.df = df
        self.gender = gender
        self.age = age
        self.education = education
        self.ethnicity = ethnicity
        self.links = self.get_links()

    # Gets the links of the group, if a field is None it will ignore the field
    def get_links(self):

        df = self.df

        if self.gender!=None:
            df = df[df['geslacht_src'] == self.gender]
            
        if self.age != None:
            df = df[df['lft_src'] == self.age]
            
        if self.education!= None:

            df = df[df['oplniv_src'] == self.education]

        if self.ethnicity != None:
            df = df[df['etngrp_src'] == self.ethnicity]
        
        return df

    # Other link values         
    def sum_n(self):
        return sum(self.links['n'])

    def ethnicity_links(self):
        return list(self.links['etngrp_dst'])
    
    def gender_links(self):
        return list(self.links['geslacht_dst'])
    
    def education_links(self):
        return list(self.links['oplniv_dst'])

    def age_links(self):    
        return list(self.links['lft_dst'])
    
def make_plots(category, name):
    '''
    Makes heatmaps of links between groups
            
    category: 'huishouden','werkschool',  'familie','buren'

    name: 'etngrp','geslacht', 'oplniv', 'lft'
    '''

    # Import dataframes and initialize Index and columns
    df = pd.read_csv(f"Data/tab_{name}.csv")   
    df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')

    Index= sorted([i for i in pd.unique(df[category+'_src'])])
    Cols = sorted([i for i in pd.unique(df[category+'_dst'])])

    # Initialize values and normalized, which can be False or True
    values = []
    normalized = False

    # Goes through columns and indexes
    for i in Index:
        value = []
        for c in Cols:

            # Sums up the amount of connections 2 groups have
            x = (np.sum((df[(df[category+'_src'] == i) & (df[category+'_dst'] == c)]))['n'])

            # If it is normalized 
            if normalized:
                x_n = np.sum(df_n[df_n[category] == c]['n'])
                value.append(x/x_n)
            else:
                value.append(x)
            
        # Append values to list
        values.append(value)

    
    # Make df with values
    df = pd.DataFrame(values, index= Index, columns= Cols)
    

    df = (df.div(df.sum(axis=1), axis=0))

    print(df)


    sns.heatmap(df.T, annot=True, fmt='.2f', cmap="Blues", cbar=True)
    plt.xlabel(f'src_{category}')
    plt.ylabel(f'dst_{category}')
    plt.tight_layout()

    # plt.show()

    if normalized:
        plt.savefig(f'Figures/tab_{name}/Heatmap/hm_normalized_{category}.jpg')
    else:
        plt.savefig(f'Figures/tab_{name}/Heatmap/hm_{category}.jpg')
    plt.show()


def get_statistics_homophily(category, name):
    '''
    Looks at the amount of connections between groups, and computes a t-test
            
            
    category: 'huishouden','werkschool',  'familie','buren'

    name: 'etngrp','geslacht', 'oplniv', 'lft'


    '''
    df = pd.read_csv(f"Data/tab_{name}.csv")    
    Index= sorted([i for i in pd.unique(df[category+'_src'])])
    Cols = sorted([i for i in pd.unique(df[category+'_dst'])])

    values = []
    for i in Index:
        value_l = []
        for c in Cols:
            x = df[(df[category+'_src'] == i) & (df[category+'_dst'] == i)]['n']
            y = df[(df[category+'_src'] == i) & (df[category+'_dst'] == c)]['n']

            t_test =  (stats.ttest_ind(x, y))
            
            last_value = ((np.mean(x)), (np.mean(y)), (t_test[1]))

            value_l.append(last_value)

        values.append(value_l)

    df = pd.DataFrame(values, index= Index, columns= Cols)

    df.to_csv(f'Data/Descriptives/tab_{name}/{category}_ttest_homophily.csv')


def get_distributions_connections(category, name):
    '''
    Returns a plot with the distributions of a certain category

    category: 'huishouden','werkschool',  'familie','buren'

    name: 'etngrp','geslacht', 'oplniv', 'lft'

    '''
    df1 = pd.read_csv(f"Data/tab_familie.csv")    
    df2 = pd.read_csv(f"Data/tab_buren.csv")    
    df3 = pd.read_csv(f"Data/tab_werkschool.csv")    
    df4 = pd.read_csv(f"Data/tab_huishouden.csv")    

    df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')

    c = 'etngrp'
    in_out = 'dst'

    
    # for i in np.sort(df[f'{c}_{in_out}'].unique()):
    x2 = 0
    for i in np.sort(df1[f'{c}_{in_out}'].unique()):
        x = 0
        print(i)
        for df in [df1, df2, df3, df4]:
            x += round(df[df[f'{c}_{in_out}'] == i].n.sum(),3)


            x2 += round(df[df[f'{c}_{in_out}'] == i].n.sum(),3)
        # print(round(df[df[f'{c}_{in_out}'] == i].n.sum()/df_n[df_n[c] == i].n.sum(),3))
        print(round(x/100000, 3))

        print(round(x2))
        # print()
    # print(df.n.sum()/df_n.n.sum())
    exit()

    person_links = Person_links(df, None,None,None,None)
    # sns.barplot(data =person_links.links, x=category+'_src', y='fn', hue = category+'_dst', estimator=np.sum)
    category = 'lft'
    category2 = 'etngrp'
    Index= sorted([i for i in pd.unique(df[category+'_src'])])
    
    Index2= sorted([i for i in pd.unique(df[category2+'_src'])])
    df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')
    

    new_df = pd.DataFrame()
    n_values = []
    age = []
    etn = []
    
    normalized = False
    if normalized:
        
        for i in Index:
            values_n_df = sum(df_n[df_n[category] == i]['n'])
            n_value = sum(df[df[category+'_src'] == i]['n'])

            new_value = round(n_value/values_n_df)
            print(int(round(new_value)))
            n_values.extend([i] * new_value)
    else:
        for i in Index2:
        
            df2 = df[df[category+'_src'] == i]

            df3 = df_n[df_n[category] == i]
    

            for j in Index:
                # print(sum(df3['n'])/sum(df2['n']))

                x = sum(df3[df3[category2] == j]['n'])

                if x:
    
                    n_value  = sum(df2[df2[category2+'_src'] == j]['n'])

                    new_value = round(n_value/x)

                    n_values.extend([i + j] * new_value)
                    age.extend([i] * new_value)
                    etn.extend([j]*new_value)

        
    new_df['Category'] = n_values
    new_df['Age'] = age
    new_df['Ethnicity'] = etn

    # Specific plot

    sns.histplot(data=new_df,  stat="count", multiple="stack",
            x="Ethnicity", kde=False,
            palette="pastel", hue="Age",
            element="bars", legend=True)

    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show() 
    plt.savefig(f'Figures/tab_{name}/{category}Distribution_normalized_concat.jpg')
    # plt.show()

    sns.histplot(data=new_df, y = 'Category')

    # plt.xticks(rotation=90)
    plt.yticks(fontsize = 8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Figures/tab_{name}/{category}Distribution_normalized_concat1.jpg')
    # plt.show()
    if normalized:
        plt.savefig(f'Figures/tab_{name}/{category}Distribution_normalized.jpg')

    else:
        plt.savefig(f'Figures/tab_{name}/{category}Distribution.jpg') 
    plt.close()



def get_distributions_n():

    df = pd.read_csv('Data/tab_n_(with oplniv).csv')

    c = 'lft'
    for i in df[c].unique():
        print(i)
        print(df[df[c] == i].n.sum())

    exit()

    for column in df.columns[0:-1]:
        new_df = pd.DataFrame()

        n_values = []
        for value in sorted(pd.unique(df[column])):
            n_value = sum(df[df[column] == value]['n'])  
            n_values.extend([value] * n_value)
        
        new_df['Category'] = n_values

        sns.histplot(data = new_df ,x='Category')
        plt.tight_layout()

        
        plt.savefig(f'Figures/tab_n/{column}.jpg')    # df.to_csv(f'Data/Descriptives/tab_{name}/{category}_ttest_homophily.csv')
        plt.show()
        plt.close()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def make_distributions_per_group(name):
    '''
    Makes a distribution of all 240 groups
    '''

    # Loads 
    df = pd.read_csv(f"Data/tab_{name}.csv")   
    df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')
    distribution_df = pd.DataFrame()
    values_df = pd.DataFrame()

    values = {}

    groups = []
    size_household = []
    for i in df.itertuples():
        values[i[1], i[2], i[3], i[4]] = 0

    for i in values:
        print(i)
        value_n = df_n[(df_n['geslacht'] == i[0]) & (df_n['lft'] == i[1]) & (df_n['etngrp'] == i[3])& (df_n['oplniv'] == i[2])]
        
        value_c = df[(df['geslacht_dst'] == i[0]) & (df['lft_dst'] == i[1]) & (df['etngrp_dst'] == i[3])& (df['oplniv_dst'] == i[2])]

        size_household.append(value_c['n'].sum()/ value_n['n'].sum())
        groups.append(str(i))

    size_household = np.array(size_household)[np.array(size_household) < 9]

    print(np.mean(size_household))

    plt.hist(size_household, bins = 8)

    plt.show()
    # exit()
    # dst_values = []
    # d_values = []
    # values = []

    # for i in df.iterrows():
    #     geslacht = i[1]['geslacht_dst']
    #     lft = i[1]['lft_dst']
    #     oplniv = (i[1]['oplniv_dst'])
    #     etngrp = i[1]['etngrp_dst']
    #     dst_group = f'{geslacht}, {lft}, {oplniv}, {etngrp}'
        

        
    #     value = df_n[(df_n['geslacht'] == geslacht) & (df_n['lft'] == lft) & (df_n['etngrp'] == etngrp)& (df_n['oplniv'] == oplniv)]


        
        
    #     if sum(value['n']) != 0:
    #         value = round(i[1]['n'])/value['n']
    #         values.append(value)
    #         # print(value)
    #         d_values.append(dst_group)
    #         dst_values.extend([dst_group] * int(float(value) * 100))

        
    # distribution_df['destination'] = dst_values
    # values_df['destination'] = d_values
    # values_df['values'] = values

    # values_df = values_df.groupby(by='destination').sum().reset_index()
    # # plt.yticks(fontsize = 5)
    # # plt.tight_layout()
    # print(values_df)

    # sns.barplot(data = values_df, x='destination', y='values')

    # plt.show()
    # values_df.to_csv('values_df.csv')
def normalized_links(df):
    gender, age, etn, edu = get_groups()

if __name__ == '__main__':
    # Takes name of tab_ as input and reads csv
    name = 'buren'
    group = 'etngrp'
    df = pd.read_csv(f"Data/tab_{name}.csv")


    # Ask for exporting descriptive statistics
    export_mean_columns = False
    export_count_values = False
    homophily_statistis = False
    distributions_n = False
    distribution_connections = True
    total_distribution = False



    # Calling functions
    if export_mean_columns:
        mean_columns(df, name)

    if export_count_values:
        count_values(df, name)



    if distributions_n:
        get_distributions_n()    

    if distribution_connections:
        get_distributions_connections(name, group)

    # get distributions(name, i) can also be filled in or make_pot(name,i)
    if homophily_statistis:
        for name in ['huishouden','werkschool',  'familie','buren']:
            for i in ['etngrp','geslacht', 'oplniv', 'lft']:
                #  get_statistics_homophily(name, i)
                # get_distributions_connections(i, name)

                make_plots(i,name)
    
    if total_distribution:
        make_distributions_per_group(name)

    
    # edges = get_total_edges(name)
    # nodes = get_total_nodes()

    # print(edges/nodes)




