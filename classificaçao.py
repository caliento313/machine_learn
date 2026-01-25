import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

df = pd.read_excel('data/cerveja_nota.xlsx')
#print(df)
df['aprovado'] = (df['nota'] > 5).astype(int)
print(df)
#plt.show()
reg = linear_model.LogisticRegression(penalty=None,fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_prob = reg.predict_proba(df[['cerveja']].drop_duplicates())
print(reg_predict)

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_prob = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())

arvore_d2 = tree.DecisionTreeClassifier(max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_prob = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado']) 
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_prob = nb.predict_proba(df[['cerveja']].drop_duplicates())


plt.plot(df['cerveja'], df['aprovado'], 'o')
#plt.show()
plt.xlabel('Nota da Cerveja')   
plt.ylabel('Aprovado (1) / Reprovado (0)')
plt.title('Classificação de Cerveja')
plt.grid(True)
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='tomato')
plt.plot(df['cerveja'].drop_duplicates(), reg_prob[:, 1], color='red')
#plt.plot(df['cerveja'].drop_duplicates(), arvore_full_predict, color='green')
#plt.plot(df['cerveja'].drop_duplicates(), arvore_full_prob[:, 1], color='purple')  
plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict, color='orange')
plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_prob[:, 1], color='brown') 
plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='pink')
plt.plot(df['cerveja'].drop_duplicates(), nb_prob[:, 1], color='gray')
plt.plot(df['cerveja'], df['aprovado'], 'o',color='blue')
plt.legend(['observações',
            #'reg predict',
            #'reg prob',
            'arvore predict',
            'arvore prob',
            'arvore d2 predict',
            'arvore d2 prob',
            'nb predict',
            'nb prob'])
plt.hlines(0.5,xmin=1, xmax=9, color='black', linestyle='--')
plt.show()
# %