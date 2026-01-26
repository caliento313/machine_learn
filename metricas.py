#%%
import pandas as pd 
from sklearn import tree
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn import linear_model

# leitura do arquivo excel
df = pd.read_excel("data/DadosComunidade2.xlsx") 
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
#print(df.info())

#tranformou em 0 e 1 as respostas sim e não
df = df.replace({'Sim':1, 'Não':0} )
#print(df.head())    
#print(df.columns)

#lista de variaveis numericas
num_vars = ['Curte games?',
            'Curte futebol?',
            'Curte livros?', 
            'Curte jogos de tabuleiro?',
            'Curte jogos de formula 1?', 
            'Curte jogos de MMA?',
            'Idade']

#lista de variaveis categoricas para dummies
dummy_vars=['Como conheceu o Teo Me Why?',
'Quantos cursos acompanhou do Teo Me Why?',
'Estado que mora atualmente',
'Formação',
'Tempo que atua na area de dados',
 'Posição da cadeira (senioridade)',
 ]

#aplicou o get_dummies transformando as variaveis categoricas em variaveis dummies
dammy = pd.get_dummies(df[dummy_vars]).astype(int)
dammy[num_vars] = df[num_vars].copy()
dammy['pessoa feliz'] = df['Voce se considera uma pessoa feliz?'].copy()
print(dammy)

# separação das variaveis preditoras e variavel alvo
features = dammy.columns[:-1].tolist()
X = dammy[features]
y = dammy['pessoa feliz']  


#Treinamento dos modelos
#modelo de Árvore de Decisão
arvore = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
arvore.fit(X, y)
arvore_predict = arvore.predict(X)
print(arvore_predict)
#modelo de Naive Bayes
naive = naive_bayes.GaussianNB()
naive.fit(X, y)
naive_predict = naive.predict(X)
print(naive_predict) 
#modelo de Regressão Logística  
reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(X, y)
reg_predict = reg.predict(X)
print(reg_predict) 


#Criação do DataFrame para comparação dos resultados
df_predict = dammy[['pessoa feliz']].copy()
df_predict['predict_arvore'] = arvore_predict
df_predict['predict_proba'] = arvore.predict_proba(X)[:,1]  
print(df_predict)
df_predict['predict_naive'] = naive_predict
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]
#print(df_predict)
df_predict['predict_reg'] = reg_predict
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]
#print(df_predict)

#Cálculo da acurácia manualmente
#md = ( df_predict['pessoa feliz'] == df_predict['predict_arvore']).mean()
#print(md)
#md2 = pd.crosstab(df_predict['pessoa feliz'], df_predict['predict_arvore'])
#print(md2)
# Cálculo das métricas com sklearn
acc_arvore = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
#print(acc_arvore)
precisao_arvore = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
#print(precisao_arvore)
recall_arvore = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
#print(recall_arvore)
roc_arvore = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['predict_proba'])
#print(roc_arvore)
auc_arvore = metrics.auc(roc_arvore[0], roc_arvore[1])
print(auc_arvore)


#Cálculo das métricas do Naive Bayes
acc_naive = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
#print(acc_naive)
precisao_naive = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
#print(precisao_naive)
recall_naive = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
#print(recall_naive)
roc_naive = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['proba_naive'])
#print(roc_naive)
auc_naive = metrics.auc(roc_naive[0], roc_naive[1])
print(auc_naive)  


#Cálculo das métricas da Regressão Logística
acc_reg = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
#print(acc_reg)
precisao_reg = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
#print(precisao_reg)
recall_reg = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
#print(recall_reg)
roc_reg = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['proba_reg'])
#print(roc_reg)
auc_reg = metrics.auc(roc_reg[0], roc_reg[1])
print(auc_reg)  

#Plotagem da Curva ROC  
plt.figure(dpi=100)
plt.plot(roc_arvore[0], roc_arvore[1],'o-')
plt.plot(roc_naive[0], roc_naive[1],'o-')
plt.plot(roc_reg[0], roc_reg[1],'o-')
plt.grid(True)
plt.title('Curva ROC - Árvore de Decisão')
plt.title('Curva ROC - Naive Bayes')
plt.xlabel('1-Especificidade (FPR)')
plt.ylabel('recall (TPR)')
plt.legend( ['AUC Árvore: %.2f'%auc_arvore, 'AUC Naive: %.2f'%auc_naive, 'AUC Regressão Logística: %.2f'%auc_reg] )
plt.show()

#salvando o modelo de regressão logística   
pd.Series({'model':reg, 'features':features}).to_pickle('data/reg_model.pkl')


# %%
