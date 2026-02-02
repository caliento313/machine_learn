# %%
import pandas as pd
# %%
df = pd.read_excel("data/cerveja_nota.xlsx")
print(df)
# %%  
from sklearn import linear_model 
from sklearn import tree 

X = df[['cerveja']] #isso é uma matriz(dataframe)
Y = df['nota']    #isso é um vetor (series)
# isso é aprendizado de máquina supervisionado

reg = linear_model.LinearRegression()  
reg.fit(X, Y)
# %%
a, b = reg.coef_[0], reg.intercept_
print(a,b)
# %%
predict = reg.predict(X.drop_duplicates())
#print(predict)
arvore_full=tree.DecisionTreeRegressor(random_state=2, max_depth=2)
arvore_full.fit(X,Y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())
arvore_d2=tree.DecisionTreeRegressor(random_state=2, max_depth=2)
arvore_d2.fit(X,Y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
#print(predict_arvore_full)
# %%

import matplotlib.pyplot as plt
plt.plot(X['cerveja'], Y, 'o')
plt.grid(True)
plt.title('Relação entre nota e cerveja')
plt.xlabel('Cerveja')
plt.ylabel('Nota')  
plt.plot(X.drop_duplicates(), predict, color='red')
plt.plot(X.drop_duplicates(), predict_arvore_full, color='green')
plt.plot(X.drop_duplicates(), predict_arvore_d2, color='orange')
#plt.show()  
# %%
tree.plot_tree(arvore_d2,feature_names=['cerveja'],filled=True)
plt.figure(figsize=(10, 6), dpi=400)
plt.show()


