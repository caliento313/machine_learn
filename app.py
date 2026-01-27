#%%
#criação do app com streamlit
import streamlit as st
import pandas as pd

#carregamento do modelo treinado
model = pd.read_pickle('data/reg_model.pkl')

# Título e descrição do aplicativo
st.title("Hello World App")
st.markdown("# Descubra a Felicidade!")
st.markdown("Este aplicativo utiliza modelos de Machine Learning para prever se uma pessoa é feliz com base em suas respostas a um questionário.")

# Coleta de dados do usuário
redes = ["Por meio de um amigo", 'LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos',
       'Twitter / X', 'Outra rede social']
st.selectbox("Como conheceu o Teo Me Why?", options=redes)

cursos_opt = ['0','1','2','3','mais de 3']
cursos = st.selectbox("Quantos cursos acompanhou do Teo Me Why?", options=cursos_opt)

estado_opt = ['MG', 'SC', 'SP', 'CE', 'PE', 'RJ', 'AM', 'PR', 'BA', 'PA', 'MT',
       'RS', 'DF', 'RN', 'ES', 'PB', 'GO', 'MA']
estado = st.selectbox("Estado que mora atualmente", options=estado_opt)

formacao_opt = ['Exatas', 'BiolÃ³gicas', 'Humanas']
formacao = st.selectbox("Formação", options=formacao_opt)

tempo_area_opt = ['Mais de 4 anos', 'Não atuo', 'De 1 ano a 2 anos',
       'De 0 a 6 meses', 'de 2 anos a 4 anos', 'De 6 meses a 1 ano']
tempo_area = st.selectbox("Tempo que atua na area de dados", options=tempo_area_opt)

senioridade_opt = ['Senior', 'Iniciante', 'Junior', 'Pleno', 'Gerencia',
       'Coordenação', 'Especialista', 'Diretoria', 'C-Level']
senioridade = st.selectbox("Posição da cadeira (senioridade)", options=senioridade_opt)
idade = st.number_input('Qual sua idade?', min_value=18, max_value=100, step=1)

# Divisão em colunas para melhor layout
col1, col2, col3 = st.columns(3)
with col1:
   video_game = st.radio('Você curte games?', ['sim', 'não'])
   Curte_futebol = st.radio('Você curte futebol?', ['sim', 'não'])
with col2:
   Curte_livros = st.radio('Você curte livros?', ['sim', 'não'])
   Curte_jogos_de_tabuleiro = st.radio('Você curte jogos de tabuleiro?', ['sim', 'não'])
with col3:
   Curte_jogos_de_formula_1 = st.radio('Você curte jogos de formula 1?', ['sim', 'não'])
   Curte_jogos_de_mma = st.radio('Você curte jogos de MMA?', ['sim', 'não'])

# Preparação dos dados para previsão
data ={'Como conheceu o Teo Me Why?': redes,
          'Quantos cursos acompanhou do Teo Me Why?': cursos_opt,
          'Curte games?': video_game,
          'Curte futebol?': Curte_futebol,
          'Curte livros?': Curte_livros,
          'Curte jogos de tabuleiro?': Curte_jogos_de_tabuleiro,
          'Curte jogos de formula 1?': Curte_jogos_de_formula_1,
          'Curte jogos de MMA?': Curte_jogos_de_mma, 'Idade': idade,
          'Estado que mora atualmente': estado, 
          'Formação': formacao,
          'Tempo que atua na area de dados': tempo_area, 
          'Posição da cadeira (senioridade)': senioridade,}

# Criação do DataFrame com os dados do usuário  
df = pd.DataFrame([data]).replace({'sim':1, 'não':0} )

# Preparação das variáveis dummies  
dummy_vars = ['Como conheceu o Teo Me Why?',
'Quantos cursos acompanhou do Teo Me Why?',
'Estado que mora atualmente',
'Formação',
'Tempo que atua na area de dados',
 'Posição da cadeira (senioridade)',
 ]

# Tratamento para variáveis com múltiplas seleções (se aplicável)
for col in dummy_vars:
    df[col] = df[col].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x
    )

# Aplicação do get_dummies
df = pd.get_dummies(df[dummy_vars]).astype(int)

# Garantia de que todas as colunas esperadas estão presentes    
df_template = pd.DataFrame(columns=[
'Como conheceu o Teo Me Why?_Amigos',
'Como conheceu o Teo Me Why?_Instagram',
'Como conheceu o Teo Me Why?_LinkedIn', 
'Como conheceu o Teo Me Why?_Outra rede social',
'Como conheceu o Teo Me Why?_Twitch',
'Como conheceu o Teo Me Why?_Twitter / X', 
'Como conheceu o Teo Me Why?_YouTube',
'Quantos cursos acompanhou do Teo Me Why?_0', 
'Quantos cursos acompanhou do Teo Me Why?_1',
'Quantos cursos acompanhou do Teo Me Why?_2', 
'Quantos cursos acompanhou do Teo Me Why?_3', 
'Quantos cursos acompanhou do Teo Me Why?_Mais que 3', 
'Estado que mora atualmente_AM',
'Estado que mora atualmente_BA',
'Estado que mora atualmente_CE',
'Estado que mora atualmente_DF', 
'Estado que mora atualmente_ES', 
'Estado que mora atualmente_GO', 
'Estado que mora atualmente_MA',
'Estado que mora atualmente_MG', 
'Estado que mora atualmente_MT', 
'Estado que mora atualmente_PA', 
'Estado que mora atualmente_PB', 
'Estado que mora atualmente_PE', 
'Estado que mora atualmente_PR', 
'Estado que mora atualmente_RJ', 
'Estado que mora atualmente_RN', 
'Estado que mora atualmente_RS', 
'Estado que mora atualmente_SC', 
'Estado que mora atualmente_SP', 
'Formação_BiolÃ³gicas', 
'Formação_Exatas', 
'Formação_Humanas', 
'Tempo que atua na area de dados_De 0 a 6 meses', 
'Tempo que atua na area de dados_De 1 ano a 2 anos',
'Tempo que atua na area de dados_De 6 meses a 1 ano',
'Tempo que atua na area de dados_Mais de 4 anos', 
'Tempo que atua na area de dados_Não atuo', 
'Tempo que atua na area de dados_de 2 anos a 4 anos',
'Posição da cadeira (senioridade)_C-Level',
'Posição da cadeira (senioridade)_Coordenação', 
'Posição da cadeira (senioridade)_Diretoria', 
'Posição da cadeira (senioridade)_Especialista', 
'Posição da cadeira (senioridade)_Gerencia', 
'Posição da cadeira (senioridade)_Iniciante', 
'Posição da cadeira (senioridade)_Junior',
'Posição da cadeira (senioridade)_Pleno', 
'Posição da cadeira (senioridade)_Senior',
'Curte games?', 
'Curte futebol?', 
'Curte livros?', 
'Curte jogos de tabuleiro?', 
'Curte jogos de formula 1?',
'Curte jogos de MMA?', 
'Idade',])

# Alinhamento das colunas do DataFrame do usuário com o template
df = pd.concat([df_template, df], axis=0).fillna(0).astype(int).reset_index(drop=True)

# Previsão usando o modelo carregado
proba = model['model'].predict_proba(df[model['features']])

# Exibição dos resultados
if proba[0][1] > 0.5:
    st.success(f"## Parabéns! Você é uma pessoa feliz! 🎉😊")
else:
    st.error(f"## Não se preocupe! A felicidade é uma jornada, não um destino. 🌈💪 ")

# Exibição da probabilidade
st.markdown(f"## A probabilidade de você ser uma pessoa feliz é de {proba[0][1]:.2%} !")  
#%%