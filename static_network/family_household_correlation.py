import pandas as pd
import numpy as np
import timeit

# Import household and family datasets
df_household = pd.read_csv('./Data/NW_data2/hh_test.csv')
df_family =  pd.read_csv('./Data/NW_data2/familie_test.csv')

# Get undirected matrices
undirected_house_hold = df_household.iloc[::2]
undirected_family = df_family.iloc[::2]

# Get the unique groups of both source group and destination group
source_groups = undirected_house_hold['source_group'].unique()
destination_groups = undirected_house_hold['destination_group'].unique()

# 
mx_h = undirected_house_hold.to_numpy()
mx_f = undirected_family.to_numpy()

mx_h = mx_h[np.lexsort((mx_h[:,-2],mx_h[:,-1]))]
mx_f = mx_f[np.lexsort((mx_f[:,-2],mx_f[:,-1]))]



household_dualarray = np.column_stack((mx_h[:,-2], mx_h[:,-1]))
family_dualarray = np.column_stack((mx_f[:,-2], mx_f[:,-1]))

household_dualarray = household_dualarray[np.lexsort((household_dualarray[:,1],household_dualarray[:,0]))]
family_dualarray = family_dualarray[np.lexsort((family_dualarray[:,1],family_dualarray[:,0]))]


unique_rows_h = np.unique(household_dualarray, axis=0, return_index=True)
unique_rows_f = np.unique(family_dualarray, axis=0, return_index=True)

f_connections, f_indices = unique_rows_f
h_connections, h_indices = unique_rows_h


f_indices = np.append(f_indices, mx_f.shape[0])
h_indices = np.append(h_indices, mx_h.shape[0])

i = 0
j = 0

same = 0
mx_h = np.delete(mx_h, 0, 1)  # delete second column of C
mx_f = np.delete(mx_f, 0, 1)  # delete second column of C

while i < len(f_connections):

        index = np.where(np.all(f_connections==f_connections[i],axis=1))[0]

        if len(index) < 1:
                i += 1
                continue
        
        index = index[0]
        # Look where connecitons are the same in both matrices
        
        
        # Look at the indices in the whole matrices these have
        
        # Get range of family and household
        # Look at the total amount of connections of the value
        range_f = f_indices[i + 1] - f_indices[index]
        range_h = h_indices[index + 1] - h_indices[i]

        # Replace a x percent of the values in the household dataset with the values in the Family dataset
        first_f = f_indices[i]
        first_h = h_indices[index]

        if range_h > range_f:
                second = int(range_f * 1)
                
        else:
                second = int(range_h * 1)
        
        old_values = mx_f[first_f: first_f + second]
  
        mx_f[first_f: first_f + second] = mx_h[first_h: first_h + second ]
        

        same += second
  
    
        x = np.unique(mx_f[first_f: f_indices[i+1]], axis=0)
        
        while len(x) != len(mx_f[first_f: f_indices[i+1]]):

                difference = abs(len(x) - len(mx_f[first_f: f_indices[i+1]]))

                number_of_rows = old_values.shape[0]
               
                random_indices = np.random.choice(number_of_rows, size=difference, replace=False)
                random_rows = old_values[random_indices, :]

                # print(random_indices)
                x = np.vstack((random_rows, x))

                print(x)
                print(mx_f[first_f: f_indices[i+1]])

        


        mx_f[first_f: f_indices[i+1]] = x
        i += 1 

copy = mx_f.copy()

copy[:, [-1, -2]] = copy[:, [-2, -1]]

mx_f =  np.vstack((copy, mx_f))

df = pd.DataFrame(mx_f)

df.columns = ['source_id','destination_id','source_group','destination_group']

df = df[df.duplicated(subset=['source_id','destination_id'], keep=False)]
print (df)

df.to_csv('overlap.csv', index=False)
