import pandas as pd
from devModel.eda import Eda

df = pd.read_csv(r'C:\Users\ALEJANDRO\Desktop\Projects\Virtual_Analyzer\Datas Corridas\test.csv')
eda = Eda(df)
#print(eda.summary())
#print(eda.statistics())
#print(eda.features_summary())
#print(eda.warnings())
#print(eda.features_correlations(method="phik", cols=None, p_value= False, matrix='z score'))

