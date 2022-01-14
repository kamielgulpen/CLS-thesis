from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    def isolation(df):
        print(df)
        columns = df.columns[1:-3]
        l = []
        for column in columns:
            row = df.loc[[column]]
            xi = row["Surinaams"]
            X = row[columns].sum(1)

            df2 = df[columns]
            ti = (sum(df2.sum()))


            isolation = (xi/X) * (xi/ti)
            print(isolation)
            
            l.append(float(isolation))
        print(sum(l))
        exit()
    #     # xi = 
    def isolation2(df):

        # print(df)
        columns = df.columns[1:-3]
        l = []
        for column in columns:
            row = df.loc[[column]]
            xi = int(row["Turks"])
            X = xi/(row[columns].sum(1))

            df2 = df[columns]
            ti = (sum(df2.sum()))
            # X = xi/ti
        
            
            isolation = (xi * X) / (xi)
            print(isolation)
            l.append(float(isolation))
        print(sum(l))
        exit()
        # xi = 


    def segregation(df):

        columns = df.columns[1:-3]
        segregation_index = []
        category = []
        for c in columns:    
            print(c)
            l = []
            A = sum([int(df.loc[[column]][c]) for column in columns])
            T = sum([int(df.loc[[column]][list(columns)].sum(1)) for column in columns])
            # print(A,T)


            for column in columns:
                row = df.loc[[column]]
                # print(row)
                
                ai = int(row[c])

                ti = int(row[list(columns)].sum(1))

        
                isolation = (ai/A) - ((ti)/(T))
            
                l.append(abs(isolation))

            
            print(sum(l) * 0.5)
            segregation_index.append(sum(l) * 0.5)
            category.append(c)
        new_df = pd.DataFrame()

        new_df['Category'] = category
        new_df['Segregation index']= segregation_index
        

        new_df.to_latex()
        print(new_df.to_latex(index=False))
        exit()
        

    def theils_t(row):
        l2 = []
        for cell in row:    
            xi = cell
            mu = np.mean(row)
            N = len(row)
            x = ((xi)/mu)*(np.log(xi/mu))
            # print(x)
            if xi:
                l2.append(x)
            else:
                l2.append(0)
        
        return (1/N * sum(l2)) / np.log(N)

    def Inequaity(category, name, df):
        df_n = pd.read_csv('Data/tab_n_(with oplniv).csv')
        Index= sorted([i for i in pd.unique(df[category+'_src'])])
        Columns = sorted([i for i in pd.unique(df[category+'_src'])])

        Columns.insert( 0,'Source')
        Columns.append('gini')
        Columns.append('Atkins')
        Columns.append('Theil')
        # print(Index)
        new_df = pd.DataFrame(columns=Columns)
        n_values = []
        # print(name)
        # print('\n')
        for i in Index:
            row = []
            for j in Index:
                
                # print(i,j)
                x = df[(df[category+'_src'] == i) & (df[category+'_dst'] == j)]['n'] /sum(df_n[df_n[category] == j]['n'])

                # total = df[df[category+'_src'] == i]['n']/sum(df_n[df_n[category] == j]['n'])
                # print(x)

                row.append(sum(x))
            

            
            # Mean absolute difference
            g = gini(row)
            a = atkinson.index(row)    
            t = theils_t(row)
            row.insert(0,i)
            row.append(g)
            row.append(a)
            row.append(t)
    

            new_df.loc[i] = row
        # print(name)
        # segregation(new_df)
        # print(new_df)
        # display(df)
        new_df.to_csv(f'Data/Descriptives/Theils_t/{name}_{category}2.csv')  
        # exit()
        print('\n')















if __name__=='__main__':
    # ,'werkschool', 'familie', 'huishouden'
    for name in [ 'buren', 'huishouden', 'familie', 'werkschool']:
        df = pd.read_csv(f"Data/tab_{name}.csv")    
        for i in ['oplniv','etngrp','geslacht',  'lft']:
            # make_plots(i, name, df)
            # get_statistics_homophily(i,name, df)
            # get_distributions_connections(i,name, df)
            Inequaity(i,name,df)
