
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pkt 
dataset = pd.read_csv("loan.csv")
# print (dataset.describe())

# print (dataset.head(3))

# print (dataset.shape)

# print (dataset.isnull())

# print (dataset.isnull().sum())

# #to print number of columns
# print (dataset.shape[1]) 

#give column wise percentage 
# print ((dataset.isnull().sum()/dataset.shape[0])*100) # To calculate the percentage of the missing values

#give over all summ and  percentage 
# print (dataset.isnull().sum().sum())
# print ((dataset.isnull().sum().sum()/(dataset.shape[0] + dataset.shape[1]))*100) 


#for not not null values 
# print(dataset.notnull().sum())
# print((dataset.notnull().sum().sum()/(dataset.shape[0]+dataset.shape[1]))*100)
