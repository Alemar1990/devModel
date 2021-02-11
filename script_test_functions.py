import pandas as pd
from devModel.eda import Eda

#df = pd.read_csv(r'C:\Users\ALEJANDRO\Desktop\Projects\Virtual_Analyzer\Datas Corridas\test.csv')
#eda = Eda(df)
#print(eda.summary())
#print(eda.statistics())
#print(eda.features_summary())
#print(eda.warnings())
#print(eda.features_correlations(method="phik", cols=None, p_value= False, matrix='z score'))
#print(eda.statistical_test(test='normality', alpha=0.01, cols=['Age'], analysis_indv=True))


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path).dropna()
#grouped_test2=df.groupby(['drive-wheels'])
#print(grouped_test2.get_group('fwd'))
eda = Eda(df)
print(eda.statistical_test(test='anova', alpha=0.01, cols=None, analysis_indv=False, group='drive-wheels', subgroups=['fwd', 'rwd'], feature_interest='price'))