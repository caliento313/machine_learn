# %%

import pandas as pd
df = pd.read_excel('data/cerveja2.xlsx')
print(df)
# %%
X= df[['temperatura', 'copo', 'espuma', 'cor']]
Y = df['classe']
x = X
y = Y

x= x.replace({'mud':1,'pint':0, 'sim':1, 'não':2, 'escura': 3, "clara":4})
print(x)
# %%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
# %%

# %%
import matplotlib.pyplot as plt
import sklearn.tree as tree

tree.plot_tree(model,feature_names=x.columns, class_names=model.classes_ ,filled=False)

plt.figure(figsize=(10, 6), dpi=400)
plt.show()