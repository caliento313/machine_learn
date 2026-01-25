# %%
import pandas as pd
df = pd.read_excel('data/dadostarwars.xlsx')
print(df.head())
#print(df.columns)

# %%
X = df[[ 'Massa(em kilos)', 'Estatura(cm)', 'Tempo de existÃªncia(em meses)']]
Y = df['Status ']
x = X
y = Y

# %%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
# %%

# %%
import matplotlib.pyplot as plt
import sklearn.tree as tree

#tree.plot_tree(model,feature_names=x.columns, class_names=model.classes_ ,filled=False)

#plt.figure(figsize=(10, 6), dpi=400)
#plt.show()
from sklearn import tree
tree.plot_tree(model, max_depth=3)
plt.show()
# %%
