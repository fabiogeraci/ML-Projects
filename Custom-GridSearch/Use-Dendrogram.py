import pandas as pd
import numpy as np
from mining_tools import my_tools
from testimbalance import abalone_input
from imbalance_tools import handle_imbalance
from create_dendogram import my_dendogram
x=pd.read_csv("HTRU_2.csv")
ab=abalone_input()
x=ab.encoded_Data()
dendo_data=x.astype(np.float)
md=my_dendogram()
md.initiate(dendo_data,'Dendogram for Abalone Data-set')
data=pd.read_csv('HTRU_2.csv')
y_htru=data.iloc[:,-1].values.astype(np.float)
x_htru=data.iloc[:,:-1].values.astype(np.float)
md1=my_dendogram()
md1.initiate(x_htru,'Dendogram for Pulser dataset')