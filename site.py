
import flask


# Create the application.
APP = flask.Flask(__name__)


# coding: utf-8

# In[5]:

import matplotlib
# coding: utf-8

#%%
#Initial Declarations

import os
from pandas import *
import numpy as np
#import statsmodels.api as sm
#from scipy.stats import norm

#import datetime


#%%
#Import and Prepare Data

df = read_csv('DataCC.csv', index_col=0)


# In[6]:



#compute returns
df_return=df/df.shift(1)-1
df_return=df_return.ix[1:]

# Retrieve Libor 
tm_USD_libor=0.0


#%%
#Library of Target Functions

#Functions that Returns an Array of Scores, one score per security.
#Argument of the function must be the dataframe with stocks returns.
#Other parameters specific to the function are defined out of function as
#global variables.

#1: Momentum Strategies
#1.1 Momentum MSCI Methodology

#Specific Parameters:
number_of_months_1=6
number_of_months_2=3

#Function
def MSCI_Momentum(df):
           
    #compute annualized vol
    vol=((np.var(df)**0.5)*(len(df)**0.5))
    
    #transform return dataframe
    df_t=df+1-tm_USD_libor/len(df)
    
    #compute momentum for the two specified windows
    six_m_m=np.prod(df_t.tail(number_of_months_1*len (df)/12))-1
    six_m_m=np.nan_to_num(six_m_m)
    three_m_m=np.prod(df_t.tail(number_of_months_2*len (df)/12))-1
    three_m_m=np.nan_to_num(three_m_m)
    
    #risk constrained momentum
    rc_six_m_m=(six_m_m/vol).to_frame()
    rc_six_m_m = rc_six_m_m.iloc[:,0].map(lambda x: 3.0 if x>3.0 else x)
    rc_six_m_m = rc_six_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    rc_three_m_m=(three_m_m/vol).to_frame()
    rc_three_m_m = rc_three_m_m.iloc[:,0].map(lambda x: 3.0 if x>3.0 else x)
    rc_three_m_m = rc_three_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    
    #calcul of momentum scores
    z_score=rc_six_m_m*0.5+rc_three_m_m*0.5
    z_df=z_score.to_frame().dropna()
    z_df.columns = ['z_score']
    momentum_z_score_df = z_df.iloc[:,0].map(lambda x: x+1 if x>0 else (1-x)**(-1)).to_frame()
    momentum_z_score_df.columns = ['momentum_z_score']
    
    return momentum_z_score_df
    

#2: Carry Strategies
    
    #...
    
    
#%%
#Library of Constraint Functions
    
#Takes returns dataframe and weight vector as arguments.
#Returns a single number.
    
#Note: for the Constrained Optimisation is much safer to provide the first
# derivative (Jacobian) of both target function and constraint function.
# Thus, when adding a new function in the library it is a good idea to provide
# the derivative, at least if it is straightforward to compute.

#1. Volatility 

def index_vol(df,x):
    
    #Compute CovMatrix
    Cov_Matrix=np.cov(df, rowvar=0)
    Cov_Matrix=np.nan_to_num(Cov_Matrix)
    
    return np.dot(np.dot(x,Cov_Matrix),np.transpose(x))

def index_vol_jacobian(df,x):
    
    #Compute CovMatrix
    Cov_Matrix=np.cov(df, rowvar=0)
    Cov_Matrix=np.nan_to_num(Cov_Matrix)
    
    return np.dot(x,Cov_Matrix*2)
    
    
    
#%%
#Optimisation Module Specifications
#Target Function must be specified.
#Takes returns dataframe as an argument.    
#Returns the weight for each security.


#Specify Target Function
target_function=MSCI_Momentum

# Specify Method of Optimisation

#Method="Ranking"
Method="Constrained"

#Specify Additional Parameters (Ranking) 
nbr_sec=5

#For Constrained Optimisation:
#1. Specify Constraint Function
constraint_function=index_vol

#2. Specify Jacobians of Target Function and Constraint Function
#(Optional, but highly recommended as it makes the optimisation more likely to work)
# It would be nice to automatise this passage (that is, to make the program select
# automatically the corresponding jacobian, if available, but I don't know how to 
# do it)

#Note: in the current version the jacobian is in fact mandatory.

constraint_jacobian=index_vol_jacobian

#Specify Additional Parameters (Constrained)
Max_Vol=0.1

Max_Weight_Allowed=0.2 #Could also be specified as minimum number of security in 
# index (just its inverse)

#Terminal Operations
#Convert Vol Specification (automatic, requires no input)
daily_max_var=Max_Vol**2.0/256.0


#%%
#Index Selection Function
#This is the core of the program

def optimal_weights(df):

    if Method=="Ranking":    
    
        #compute scores
        scores=MSCI_Momentum(df)
        
        #rank z scores 
        scores["Ranking"] = scores['momentum_z_score'].argsort()
        
        #generate weights
        scores["Weights"] = scores["Ranking"].map(lambda x: 1 if x < nbr_sec else 0)
        scores["Weights"] =scores["Weights"]/np.sum(scores["Weights"])    
        Composition=Series(scores["Weights"],index=scores["Weights"].index)
        
    if Method=="Constrained":
        
        from scipy.optimize import minimize
        
        #Compute Score
        scores=MSCI_Momentum(df)
        
        #Initiate Weight Array
        Poids_0=np.ones(len(scores))
        Poids_0=Poids_0/np.sum(Poids_0)
             
        #Define Target
        def target_fun(x):
            total_score=-np.dot(x,scores)
            return total_score
        def target_fun_derivative(x):
            return -scores
        
        #Define Constraints
        Constraints=({'type': 'ineq',
        'fun' : lambda x : daily_max_var-constraint_function(df,x),
        'jac' : lambda x : -constraint_jacobian(df,x)},
        {'type' : 'eq', 
         'fun' : lambda x : np.sum(x)-1,
        'jac' : lambda x : np.ones(len(x))
        })
        
        #Define minimum value for weights (always 0) and maximum (in range (0,1] )
        bnds=[(0,Max_Weight_Allowed)]*len(Poids_0)
              
        #Define Target Function
        res=minimize(target_fun, Poids_0, #jac = target_fun_derivative,
                   constraints=Constraints, method="SLSQP",
                   bounds=bnds,
                   options={'disp': True})
        
        #Extract Weights
        Poids=res.x
            
        #Get Rid of Super Small Poids
        f = lambda x : x if x>10**-10 else 0.0
        f=np.vectorize(f)
        Poids=f(Poids)
   
        #Normalise
        Poids=Poids/np.sum(Poids)
        Composition=Series(Poids,index=scores.index)
    
    return Composition


#%%
#BackTesting Module

#t: how many months ago to start the test. 6 Months default

#Note: at the moment the rebalancing frequency is fixed to 20 trading days.
def back_test(df_return,t=6):
    
    u=len(df_return)-t*20    
    df_return_bt=df_return.ix[:-t*20]
                      
        #Compute Optimal Composition
    Poids_bt=optimal_weights(df_return_bt)
                    
        #compute returns
    next_returns=df_return.ix[u:].fillna(0)
    next_20_returns=next_returns.head(20)
    optimum_return=np.dot(next_20_returns,Poids_bt)
    return_series=Series(optimum_return,index=next_20_returns.index)
    
    for j in range(t-1,0,-1):
            
        s=len(df_return)-j*20    
        df_return_bt=df_return.ix[:-j*20]
                      
        #Compute Optimal Composition
        Poids_bt=optimal_weights(df_return_bt)
                    
        #compute returns
        next_returns=df_return.ix[s:].fillna(0)
        next_20_returns=next_returns.head(20)
        optimum_return=np.dot(next_20_returns,Poids_bt)
        return_series_prov=Series(optimum_return,index=next_20_returns.index)
        return_series=np.hstack([return_series,return_series_prov])
        
    return_series_date=Series(return_series,index=df_return.ix[u:].index)    
    
    base_1_backtest=np.ones(len(return_series_date)+1)
    for i in range(1,len(base_1_backtest)):
        base_1_backtest[i]=base_1_backtest[i-1]*(1+return_series_date[i-1])
     
    base_1_backtest_date=Series(base_1_backtest,index=df_return.ix[u-1:].index) 
    
    return base_1_backtest_date
    
    
#%%
#Plotting Module (for quick tests within Python)




# In[7]:

Sd=back_test(df_return)
type(Sd)


# In[8]:

df=df[:-1]
df_return=df_return[:-1]


# In[9]:

import matplotlib.pyplot as plt    
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
a=back_test(df_return)
df["S&P_100"] = df.sum(axis=1)
df["S&P_100 Base 1"]=df["S&P_100"]/df["S&P_100"][len(df["S&P_100"])-len(a)]
b=df["S&P_100 Base 1"].tail(len(a))
sp=b.to_frame()
sp.columns=["values"]
sp=sp.reset_index()
sp=sp['values']+1
print sp
#sp=sp.drop('Date',1)

b=a.to_frame()

# In[11]:

b.columns=["values"]


# In[12]:

b=b.reset_index()


# In[13]:

c=b['values']


# In[15]:

H=b.drop('Date',1)


# In[17]:

import json


# In[18]:

s=H.to_json(date_format='iso',orient='split')

d=json.loads(s)

T=json.dumps([{"x": date, "y": val} for date, val in zip(d['index'], d['data'])])

l=sp.to_json(date_format='iso',orient='split')
print l
q=json.loads(l)
sp=json.dumps([{"x": date, "y": val} for date, val in zip(q['index'], q['data'])])
print sp

import pygal
import json
from urllib2 import urlopen  # python 2 syntax

from pygal.style import DarkSolarizedStyle


@APP.route('/')
def index():
    bar_chart = pygal.Line(height=420 ,weight=200)

    bar_chart.title = "Your Index"
    bar_chart.x_labels = map(str, range(11))
    bar_chart.add('Index', H["values"])
    #bar_chart.add('Padovan', [1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 12]) 
    chart = bar_chart.render(is_unicode=True)
    #data=T
    return flask.render_template('demo.html', my_data=T,my_datas=sp)



if __name__ == '__main__':
    APP.debug=True
    APP.run()