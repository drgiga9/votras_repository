

import streamlit as st
import fonctions
import os
from dotenv import load_dotenv, find_dotenv
import pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings



#passwordvalue = os.getenv("Password") .  // code type pour récupérer la clé API sur Heroku

#load_dotenv(find_dotenv(), override=True)    # code pour récupérer la clé API depuis le fichier .env
#api_key = os.environ.get('OPENAI_API_KEY')   # code pour récupérer la clé API depuis le fichier .env

api_key  = os.getenv("OPENAI_API_KEY")   # code pour récupérer la clé API OPENAI sur Heroku

api_key_pinecone  = os.getenv("PINECONE_API_KEY")   # code pour récupérer la clé API PINECONEsur Heroku
environment_pinecone  = os.getenv("PINECONE_ENV")   # code pour récupérer l'environnement Pinecone sur Heroku

#pinecone.init(api_key=api_key_pinecone, environment=environment_pinecone) # code pour initialiser Pinecone sur Heroku
#pc = pinecone.Pinecone( api_key=api_key_pinecone, environment=environment_pinecone)



if not api_key:
    st.error("Clé API OpenAI non trouvée. Veuillez vérifier vos variables d'environnement.")
    raise EnvironmentError("Clé API OpenAI requise non configurée.")

if not api_key_pinecone:
    st.error("Clé API Pinecone non trouvée. Veuillez vérifier vos variables d'environnement.")
    raise EnvironmentError("Clé API Pinecone requise non configurée.")

if not environment_pinecone:
    st.error("Environnement Pinecone non trouvé. Veuillez vérifier vos variables d'environnement.")
    raise EnvironmentError("Environnement Pinecone requis non configuré.")

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

index_name = 'askadocument'


try:
    pc = pinecone.Pinecone( api_key=api_key_pinecone, environment=environment_pinecone)
    if index_name in pc.list_indexes().names():
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
except Exception as e:
    st.error("Erreur lors de la connexion à Pinecone ou lors de la manipulation de l'index.")
    st.error(str(e))
    raise


if __name__ == "__main__":
    

    
    st.subheader('Notre Chatbot Expert en Assurances🤖')
    st.session_state.vs = vector_store
    
    with st.sidebar:
        st.image('img.webp')
        st.subheader('A propos de ce Chatbot')
        st.write('Ce Chatbot est entrainé sur le Code des Assurances pour répondre à vos questions.')

    # if there's no chat history in the session state, create it
    if 'history' not in st.session_state:
        st.session_state.history = ''

    # if there's no question in the session state, create it
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ''

    if st.session_state.question_input: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs

            answer = fonctions.ask_and_get_answer(vector_store, st.session_state.question_input)

            # text area widget for the LLM answer
            st.text_area('Votre réponse : ', value=answer)

            st.divider()

            # the current question and answer
            value = f'Q : {st.session_state.question_input} \nR : {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'

            # clear the question input
            st.session_state.question_input = ''

    # text area widget for the chat history
    if st.session_state.history == '':
        st.text_area(label='Historique de la conversation', key='history', height=200)
    else:
        st.text_area(label='Historique de la conversation', key='history', height=250)

    # user's question text input widget
    st.text_input('Posez votre question :', value=st.session_state.question_input, key='question_input')



# run the app: streamlit run ./chat_with_documents.py


