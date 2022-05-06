from matplotlib.pyplot import xlim
import pandas as pd
import numpy as np
import timeit
import random

# Import household and family datasets
df_household = pd.read_csv('./Data/NW_data/hh_test2.csv')
df_family =  pd.read_csv('./Data/NW_data/familie_test.csv')

# Get undirected matrices
undirected_house_hold = df_household.iloc[::2]
undirected_family = df_family.iloc[::2]

# Get the unique groups of both source group and destination group
source_groups = undirected_house_hold['source_group'].unique()
destination_groups = undirected_house_hold['destination_group'].unique()

# Get the undirecte matrices
mx_h = undirected_house_hold.to_numpy()
mx_f = undirected_family.to_numpy()

mx_h = np.delete(mx_h, -1, 1)   


# Ordering household and family matrices based on source group and destination group
# First kn source group than n destiantion grou
mx_h = mx_h[np.lexsort((mx_h[:,-2],mx_h[:,-1]))]
mx_f = mx_f[np.lexsort((mx_f[:,-2],mx_f[:,-1]))]

mx_h = mx_h[np.lexsort((mx_h[:,-1],mx_h[:,-2]))]
mx_f = mx_f[np.lexsort((mx_f[:,-1],mx_f[:,-2]))]

# Make an matrix from only the source group and destination group
household_dualarray = np.column_stack((mx_h[:,-2], mx_h[:,-1]))
family_dualarray = np.column_stack((mx_f[:,-2], mx_f[:,-1]))


# Sort first on destination and then on source group
# household_dualarray = household_dualarray[np.lexsort((household_dualarray[:,0],household_dualarray[:,1]))]
# family_dualarray = family_dualarray[np.lexsort((family_dualarray[:,0],family_dualarray[:,1]))]


# Take all the unique connections between two groups for example 0,0 or 233, 209
# Get the indexes when a new unique connection starts
unique_rows_h = np.unique(household_dualarray, axis=0, return_index=True)
unique_rows_f = np.unique(family_dualarray, axis=0, return_index=True)



f_connections, f_indices = unique_rows_f
h_connections, h_indices = unique_rows_h

# Add last value to the indeces
f_indices = np.append(f_indices, mx_f.shape[0])
h_indices = np.append(h_indices, mx_h.shape[0])



i = 0
j = 0

same = 0

# print( mx_f)
mx_h = np.delete(mx_h, 0, 1)  # delete index

mx_f = np.delete(mx_f, 0, 1)  # delete sindex


print(f_indices)
print(h_indices)

new_matrix = np.array([-1,-1,-1,-1])
while i < len(f_connections):
     
        index = np.where(np.all(h_connections==f_connections[i],axis=1))[0]
      
        if len(index) < 1:

                # mx_f_ = mx_f[f_indices[i]: f_indices[i + 1]]

                # # print(mx_f_)
                # copy = mx_f_.copy()

                # copy[:, [-1, -2]] = copy[:, [-2, -1]]

                # stack = np.vstack((copy, mx_f_))

                # new_matrix = np.vstack((new_matrix, stack))
                i += 1
                
 
                continue
        
        index = index[0]
        # Look where connecitons are the same in both matrices
        
        
        # Look at the indices in the whole matrices these have
        
        # Get range of family and household
        # Look at the total amount of connections of the value
        range_f = f_indices[i + 1] - f_indices[i]
        range_h = h_indices[index + 1] - h_indices[index]

        
        # print(range_f, range_h)

        
        # Replace a x percent of the values in the household dataset with the values in the Family dataset
        first_f = f_indices[i]
        first_h = h_indices[index]

      
        # If range of h is bigger take range f otherwise take range h
        if range_h > range_f:
                second = int(range_f * 1)
                
        else:
                second = int(range_h * 1)
        
        old_values = mx_f[first_f: first_f + second]

        src, dest = old_values[0][-2:]

        old_values = mx_f[first_f: f_indices[i+1]]
        
        # mx_f_ = np.column_stack((mx_f[first_f: f_indices[i+1]][:,0], mx_f[first_f: f_indices[i+1]][:,1]))
        
        # copy = mx_f_.copy()
        # copy[:, [-1, -2]] = copy[:, [-2, -1]]

        # stack = np.vstack((copy, mx_f_))
        

        # o_shape = np.unique(stack, axis=0).shape
        # # exit()


        mx_f_ = np.column_stack((mx_f[first_f: f_indices[i+1]][:,0], mx_f[first_f: f_indices[i+1]][:,1]))
        
        copy = mx_f_.copy()

        copy[:, [-1, -2]] = copy[:, [-2, -1]]

        stack = np.vstack((copy, mx_f_))
        

        o_stack = np.unique(stack, axis=0)

   


        # print( mx_f)
        mx_f[first_f: first_f + second] = mx_h[first_h: first_h + second ]
        
        # print(mx_f[first_f: first_f + second])
        # print('\n')
        # print(mx_h[first_h: first_h + second ])
        
        
        
        mx_f_2 = mx_f[first_f: f_indices[i+1]]

        mx_f_2 = np.column_stack((mx_f_2[:,0],mx_f_2[:,1]))

        copy = mx_f_2.copy()
        copy[:, [-1, -2]] = copy[:, [-2, -1]]

        stack = np.vstack((copy, mx_f_2))
        
        
        n_stack = np.unique(stack, axis=0)
       


        # print(mx_h[first_h: first_h + second ])
        # print(mx_f[first_f: first_f + second])
        
        # print(mx_f[first_f: f_indices[i+1]].shape)

        # print(np.unique(mx_f[first_f: f_indices[i+1]], axis=0).shape)
    
        # exit()
        # print(mx_f[first_f: first_f + second])

       

        same += second
  
    
        x = np.unique(mx_f[first_f: f_indices[i+1]], axis=0)
        
        # print(len(x), len(mx_f[first_f: f_indices[i+1]]))
        # print(x)


        
        # print(mx_f_.shape)
        # exit()

  
        # a = 0
        # if o_stack.shape[0] != n_stack.shape[0]:
        #         # if a == 1:
        #         #         print(o_stack.shape[0], n_stack.shape[0])
        #         difference = int(abs(o_stack.shape[0] - n_stack.shape[0]))


               
        #         number_of_rows = int(o_stack.shape[0]/2)

        #         random_rows = o_stack[:number_of_rows]

               
             
                
                
        #         # print(random_indices)
               

        #         copy = random_rows.copy()
                
               
        #         copy[:, [-1, -2]] = copy[:, [-2, -1]]


              
        #         stack = np.vstack((random_rows, copy))
                
      
         
        #         # print(n_stack, stack)
        #         n_stack2 = np.unique(np.vstack((n_stack, stack)), axis =0)
           
        #         a +=1

        #         # n_stack = np.unique(n_stack, axis=0)

        #         index = np.intersect1d(np.unique(n_stack2.flatten()), np.unique(n_stack.flatten()), assume_unique=True, return_indices=True)[1]
        #         index2 = np.setdiff1d(range(np.unique(n_stack2.flatten()).shape[0]), index, assume_unique=True)

        #         # print(n_stack[])
                
                

           
        #         # print(n_stack)
        #         # print(n_stack2[index])
                
             
        #         n_stack = np.vstack((n_stack, n_stack2[index2[:difference]]))


       
        # x = np.zeros((n_stack.shape[0], 1))

        # y = np.zeros((n_stack.shape[0], 1))
        # x.fill(src)
        # y.fill(dest)

        # n_stack = np.hstack((n_stack,x, y))
        # print(new_matrix, n_stack)
        # new_matrix = np.vstack((new_matrix, n_stack))


                # print(mx_f[first_f: f_indices[i+1]])

 
        # mx_f[first_f: f_indices[index+1]] = x
        
 
        
        # print(np.unique(mx_f[first_f: f_indices[i+1]], axis=0).shape)
        # print(mx_f[first_f: f_indices[index+1]])
        # exit()
        i += 1 

print(mx_f.shape)

copy = mx_f.copy()

copy[:, [-1, -2]] = copy[:, [-2, -1]]
copy[:, [0, 1]] = copy[:, [1, 0]]
print('\n')

print(copy) 

print('\n')
print(mx_f)
mx_f =  np.vstack((copy, mx_f))
print(np.unique(mx_f, axis=0).shape)
print(mx_f.shape)

df = pd.DataFrame(mx_f)

df.columns = ['source_id','destination_id','source_group','destination_group']


print(df)
df.to_csv('./Data/NW_data/overlap.csv', index=False)
df = df[df.duplicated(subset=['source_id','destination_id'], keep=False)]
print (df)