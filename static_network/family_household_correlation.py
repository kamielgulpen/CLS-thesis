import pandas as pd
import numpy as np

df_household = pd.read_csv('./Data/NW_data2/huishouden_nw_barabasi=0_reciprocity_0_.csv')
df_family =  pd.read_csv('./Data/NW_data2/familie_nw_barabasi=0_reciprocity_0_.csv')

undirected_house_hold = df_household.iloc[::2]
undirected_family = df_family.iloc[::2]

source_groups = undirected_house_hold['source_group'].unique()
destination_groups = undirected_house_hold['destination_group'].unique()

print(len(source_groups), len(destination_groups))

mx_h = undirected_house_hold.to_numpy()
mx_f = undirected_family.to_numpy()

print(mx_h, mx_f)
count = 0


# for i in source_groups:
#     count +=1
#     print(count)
#     for j in destination_groups:
        
#         rows = mx[mx[:,3] == i]
        # undirected_house_hold[(undirected_house_hold['source_group'] == i) &
        #                         undirected_house_hold['destination_group'] == j]

