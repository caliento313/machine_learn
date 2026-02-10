import pandas as pd 
from sklearn import linear_model, model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from feature_engine import discretisation, encoding 
from sklearn import metrics
from sklearn import pipeline


#Metodo SEMMA

# Sample: Amostra dos dados para análise exploratória
# sample é a etapa de amostragem dos dados para análise exploratória.
# Nessa etapa, selecionamos uma amostra representativa dos dados para realizar análises iniciais,
#  identificar padrões e entender as características das variáveis.
#  A amostra pode ser selecionada de forma aleatória ou estratificada, 
# dependendo do objetivo da análise e da estrutura dos dados.

# Explore: Análise exploratória dos dados para entender as variáveis e suas relações 
# explore é a etapa de análise exploratória dos dados, onde examinamos as variáveis, 
# suas distribuições, correlações e relações com a variável resposta. Nessa etapa, 
# utilizamos técnicas de visualização, estatísticas descritivas e análise de correlação
#  para entender melhor os dados e identificar padrões ou insights que possam ser relevantes 
# para a construção do modelo preditivo.  
#  
# Modify: Modificação dos dados, tratamento de valores ausentes, criação de novas variáveis, etc.
# modify é a etapa de modificação dos dados, onde realizamos o tratamento de valores ausentes, 
# a criação de novas variáveis, a transformação de variáveis existentes e outras manipulações 
# necessárias para preparar os dados para a construção do modelo preditivo. Nessa etapa, 
# aplicamos técnicas de limpeza de dados, engenharia de features e transformação 
# de variáveis para melhorar a qualidade dos dados e facilitar a construção do modelo.

# Model: Construção do modelo preditivo utilizando técnicas de machine learning 
# model é a etapa de construção do modelo preditivo, onde aplicamos técnicas de machine learning
#  para treinar um modelo com os dados preparados. Nessa etapa, selecionamos o algoritmo de machine learning 
# mais adequado para o problema em questão, ajustamos os hiperparâmetros do modelo e treinamos
#  o modelo utilizando os dados de treinamento. O objetivo é criar um modelo que seja 
# capaz de fazer previsões precisas com base nas variáveis preditoras.

# Assess: Avaliação do modelo utilizando métricas de desempenho e validação cruzada
# assess é a etapa de avaliação do modelo, onde utilizamos métricas de desempenho e técnicas 
# de validação cruzada para avaliar a performance do modelo preditivo. Nessa etapa,
#  calculamos métricas como acurácia, precisão, recall, F1-score e AUC-ROC para medir a qualidade das previsões
#  do modelo. Além disso, aplicamos técnicas de validação cruzada para obter uma estimativa mais robusta 
# da performance do modelo e evitar overfitting.

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

#$$$$        SAMPLE 
# 
# (Seleção de uma amostra representativa dos dados para análise exploratória)
# Para a etapa de amostragem, podemos selecionar uma amostra representativa dos dados para 
# realizar a análise exploratória. Isso pode ser feito utilizando a função `sample` do pandas, 
# que permite selecionar uma amostra aleatória dos dados. Por exemplo, podemos selecionar 10% dos dados 
# para análise exploratória:

# Dividindo em treino e validação
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
    X, y, test_size=0.5, random_state=42,
)
#print(X_train)
#print(y_train)  

#print('Varialvel Resposta  Original:', y.mean())
#print('Variavel Resposta  Treinamento:', y_train.mean())
#print('Variavel Resposta  Validacao:', y_valid.mean())

#  $$$$      EXPLORE 
# (Análise exploratória dos dados, missing values, análise de variáveis preditoras)

# Análise de valores ausentes
X_train.isna().sum().sort_values(ascending=False)
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
arvores = tree.DecisionTreeClassifier( random_state=42)
arvores.fit(X_train, y_train)

# Avaliação do modelo
#plt.figure(dpi=100, figsize=(50,40))
#tree.plot_tree(arvores, feature_names= X_train.columns, filled=True, class_names=[str(i) for i in arvores.classes_])
#plt.show()

# Análise de importância das variáveis
features_importance = (pd.Series(arvores.feature_importances_, index=X_train.columns)
                       .sort_values(ascending=False).reset_index())
#print(features_importance)

# Análise acumulada da importância das variáveis
features_importance['acum'] = features_importance[0].cumsum()
features_importance[features_importance['acum'] < 0.95]
#print(features_importance[features_importance['acum'] < 0.95])

best_features = features_importance[features_importance['acum'] < 0.95]['index'].tolist()
#print(best_features)

#  $$$$      MODIFY
# (Modificação dos dados, tratamento de valores ausentes, criação de novas variáveis,)
# Discretização das variáveis preditoras utilizando árvore de decisão
# (Transformação de variáveis contínuas em categóricas para melhorar a performance do modelo

tree_discratizaito = discretisation.DecisionTreeDiscretiser(variables=best_features,regression=False, 
                                                            bin_output='bin_number',
                                                            cv=3)

onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True, drop_last=True )

#  $$$$      MODEL
# (Construção do modelo preditivo utilizando técnicas de machine learning)  

# O código apresentado tem como objetivo treinar e avaliar um modelo de Regressão Logística 
# utilizando a biblioteca scikit-learn, aplicando um fluxo padronizado de pré-processamento e modelagem 
# por meio de um Pipeline. Inicialmente, o modelo é definido sem regularização (penalty=None),
#  com controle de aleatoriedade (random_state=42) e número máximo de iterações 
# ajustado para garantir a convergência.

# O Pipeline é composto por três etapas principais: discretização das variáveis, one-hot encoding e modelagem.
#  A discretização transforma variáveis contínuas em faixas, facilitando a captura de padrões não lineares. 
# Em seguida, o one-hot encoding converte variáveis categóricas em variáveis binárias,
#  tornando-as compatíveis com o modelo. Por fim, a regressão logística é ajustada aos dados. 
# O uso do Pipeline garante que o mesmo pré-processamento seja aplicado de forma consistente às bases de treino,
# teste e out-of-time, evitando vazamento de informação.

# O treinamento do modelo é realizado exclusivamente com a base de treino. Após o ajuste,
#  o desempenho é avaliado nessa mesma base por meio da acurácia, da AUC (Area Under the Curve) e da curva ROC, 
# utilizando tanto as classes previstas quanto as probabilidades estimadas para a classe positiva.

# Em seguida, o modelo é avaliado na base de validação/teste, 
# seguindo exatamente o mesmo processo de predição e cálculo das métricas.
#  Essa etapa permite analisar a capacidade de generalização do modelo em dados não utilizados no treinamento.

# Por fim, o desempenho é mensurado na base OOT (Out-of-Time), 
# que representa dados de um período temporal distinto e posterior ao treinamento.
#  A avaliação nessa base é fundamental para verificar a estabilidade temporal do modelo 
# e identificar possíveis degradações de performance ao longo do tempo.

# Como resultado, são apresentadas as métricas de acurácia e AUC para as bases de treino, teste e OOT, 
# permitindo uma comparação direta entre elas e fornecendo subsídios para análise de overfitting,
#  generalização e robustez do modelo ao longo do tempo.

reg = linear_model.LogisticRegression(penalty= None, random_state=42, max_iter=1000)

model_pipeline = pipeline.Pipeline(steps=[
    ('discretizacao', tree_discratizaito),  
    ('onehot', onehot),
    ('modelo', reg)
])  
  
model_pipeline.fit(X_train, y_train) 
# 
X_train_predict = model_pipeline.predict(X_train)
X_train_predict_proba = model_pipeline.predict_proba(X_train)[:,1]
acc_train = metrics.accuracy_score(y_train, X_train_predict)
auc_train = metrics.roc_auc_score(y_train, X_train_predict_proba)
roc_train = metrics.roc_curve(y_train, X_train_predict_proba)
print('Acurácia Treino:', acc_train)
print('AUC Treino:', auc_train) 

X_test_predict = model_pipeline.predict(X_valid)
X_test_predict_proba = model_pipeline.predict_proba(X_valid)[:,1]   
acc_test = metrics.accuracy_score(y_valid, X_test_predict)
auc_test = metrics.roc_auc_score(y_valid, X_test_predict_proba)
roc_test = metrics.roc_curve(y_valid, X_test_predict_proba)
print('Acurácia Teste:', acc_test)
print('AUC Teste:', auc_test)

oot_predict = model_pipeline.predict(oot[features])
oot_predict_proba = model_pipeline.predict_proba(oot[features])[:,1] 
acc_oot = metrics.accuracy_score(oot[target], oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], oot_predict_proba) 
roc_oot = metrics.roc_curve(oot[target], oot_predict_proba)
print('Acurácia OOT:', acc_oot)
print('AUC OOT:', auc_oot)  

plt.figure(dpi=100, figsize=(10,10))
plt.plot(roc_train[0], roc_train[1]    )
plt.plot(roc_test[0], roc_test[1],     )
plt.plot(roc_oot[0], roc_oot[1],     )
plt.grid(True)
plt.xlabel('sensibilidade')
plt.ylabel('especificidade  ')
plt.title('ROC Curves')
plt.legend([f'Treino AUC: {100*auc_train:.4f}',
            f'Teste AUC: {100*auc_test:.4f}',
            f'OOT AUC: {100*auc_oot:.4f}'])
plt.show()