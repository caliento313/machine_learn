# %%
import pandas as pd
df = pd.read_excel('data/frutas2.xlsx')
print(df)
 # %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier()   
# %%
y = df['Fruta']
caracteristicas = df[['Arredondada', 'Suculenta', 'Vermelha', 'Doce']]
X = caracteristicas
# %%        
arvore.fit(X, y)
# %%
previsao = arvore.predict([[1, 1, 1, 1]])
print(previsao)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
tree.plot_tree(arvore, feature_names=caracteristicas.columns, class_names=arvore.classes_, filled=True)
plt.show()
# %%
probabilidades = arvore.predict_proba([[1, 1, 1, 1]])[0]
pd.Series(probabilidades, index=arvore.classes_)
# %%


