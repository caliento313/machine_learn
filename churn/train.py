import pandas as pd 
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt

# Configurações para exibição completa dos DataFrames
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
#pd.options.display.float_format = '{:,.4f}'.format
df = pd.read_csv('data/abt_churn.csv')
#print(df.head())

# Análise da variável dtRef para separar treino e oot
df['dtRef'].value_counts().sort_index()
#print(df['dtRef'].value_counts().sort_index())
oot = df[df['dtRef'] == df['dtRef'].max()].copy()
#print(oot)

# Criando conjunto de treino
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
#print(df_train)

#removendo coluna
df_train.columns
df_train.drop(columns=['dtRef'], inplace=True)
#print(df_train.columns)

# Separando variáveis preditoras e variável resposta
features = df_train.columns[2:-1].tolist()
target = 'flagChurn'
X, y = df_train[features], df_train[target]
#print(X)
#print(y) 

# Dividindo em treino e validação
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
    X, y, test_size=0.5, random_state=42,
)
#print(X_train)
#print(y_train)  

#print('Varialvel Resposta  Original:', y.mean())
#print('Variavel Resposta  Treinamento:', y_train.mean())
#print('Variavel Resposta  Validacao:', y_valid.mean())

# Análise de valores ausentes
X_train.isna().sum().sort_values(ascending=False).head(10)
#print(X_train.isna().sum().sort_values(ascending=False))

# Análise inicial das variáveis preditoras
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg([ 'mean','median']).transpose()
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1] 
sumario.sort_values(by=['diff_rel'], ascending=False)
#print(sumario.sort_values(by=['diff_abs'], ascending=False))

# Construção do modelo de Árvore de Decisão
arvores = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
arvores.fit(X_train, y_train)

#plt.figure(dpi=100, figsize=(50,40))
#tree.plot_tree(arvores, feature_names= X_train.columns, filled=True, class_names=[str(i) for i in arvores.classes_])
#plt.show()

# Análise de importância das variáveis
features_importance = (pd.Series(arvores.feature_importances_, index=X_train.columns)
                       .sort_values(ascending=False).reset_index())
print(features_importance)

# Análise acumulada da importância das variáveis
features_importance['acum'] = features_importance[0].cumsum()
features_importance[features_importance['acum'] < 0.95]
print(features_importance[features_importance['acum'] < 0.95])
