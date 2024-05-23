

import streamlit as st



#Embedding and Uploading to a vector Database (Pinecone)

def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client
    import os
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import ServerlessSpec



    #passwordvalue = os.getenv("Password") .  // code type pour récupérer la clé API sur Heroku

    #load_dotenv(find_dotenv(), override=True)    # code pour récupérer la clé API depuis le fichier .env
    #api_key = os.environ.get('OPENAI_API_KEY')   # code pour récupérer la clé API depuis le fichier .env

    api_key  = os.getenv("OPENAI_API_KEY")   # code pour récupérer la clé API OPENAI sur Heroku

    api_key_pinecone  = os.getenv("PINECONE_API_KEY")   # code pour récupérer la clé API PINECONEsur Heroku
    environment_pinecone  = os.getenv("PINECONE_ENV")   # code pour récupérer l'environnement Pinecone sur Heroku





    if not api_key:
        st.error("Clé API OpenAI non trouvée. Veuillez vérifier vos variables d'environnement.")
        raise EnvironmentError("Clé API OpenAI requise non configurée.")

    if not api_key_pinecone:
        st.error("Clé API Pinecone non trouvée. Veuillez vérifier vos variables d'environnement.")
        raise EnvironmentError("Clé API Pinecone requise non configurée.")

    if not environment_pinecone:
        st.error("Environnement Pinecone non trouvé. Veuillez vérifier vos variables d'environnement.")
        raise EnvironmentError("Environnement Pinecone requis non configuré.")


    try:
        pc = pinecone.Pinecone(api_key=api_key_pinecone, environment=environment_pinecone)       
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

        # loading from existing index
        if index_name in pc.list_indexes().names():
            print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            print('Ok')
        else:
            # creating the index and embedding the chunks into the index 
            print(f'Creating index {index_name} and embeddings ...', end='')

            # creating a new index
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
            ) 
            )

            # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
            # inserting the embeddings into the index and returning a new Pinecone vector store object. 
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
            print('Ok')
            
        return vector_store
    
    except Exception as e:
        print(f"Erreur lors de la création ou de la récupération de l'index {index_name}: {e}")
        raise


def ask_and_get_answer(vector_store, question, k=15):

    try:

        import os
        from langchain_openai import ChatOpenAI
        from langchain.prompts import (
        PromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        )
        from langchain_core.output_parsers import StrOutputParser
        from langchain.schema.runnable import RunnablePassthrough

        api_key  = os.getenv("OPENAI_API_KEY")   # code pour récupérer la clé API OPENAI sur Heroku

        review_template_str = """Votre tâche consiste à répondre à des questions sur les assurances.
        Utilisez le contexte suivant pour répondre aux questions.
        Soyez aussi détaillé que possible, mais n'inventez pas d'informations
        qui ne soit pas tirée du contexte.
        Vous ne devez pas utiliser vos propres connaissances ni faire de recherches sur le Web.
        Si vous ne connaissez pas la réponse, dites
        que vous ne savez pas, que vous etes entrainé pour répondre à des questions techniques et juridiques 
        basées sur le Code des Assurances Français, 
        et proposez à l'utilisateur de poser une nouvelle question sur ces thèmes précis.

        {context}
        """

        review_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"],
                template=review_template_str,
            )
        )

        review_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}",
            )
        )
        messages = [review_system_prompt, review_human_prompt]

        review_prompt_template = ChatPromptTemplate(
            input_variables=["context", "question"],
            messages=messages,
        )

        chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

        review_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
        )

        response = review_chain.invoke(question)
        return(response)

    except Exception as e:
        print(f"Erreur lors de la génération de la réponse à la question: {question}, {e}")
        raise






















"""
def ask_and_get_answer(vector_store, q, k=15):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate

    # Define the role for the LLM
    role = "L'assistant doit uniquement utiliser les informations contenues dans le vector_store pour répondre à la question. Il ne doit pas utiliser ses propres connaissances ou effectuer des recherches sur le web."

    # Create a prompt template with the role
    prompt_template = PromptTemplate(
        template=role + "\n\nQuestion: {q}\nRéponse:",
        input_variables=["q"]
    )

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    answer = chain.invoke(q)

    #If the AI didn't find an answer in the vector_store, return the error message
    if answer == "":
        answer = "Je suis désolé, je ne peux répondre qu'à des questions techniques et juridiques, concernant les assurances, et contenues dans le Code des Assurances Français. Veuillez poser une autre question sur ces thèmes précis."
    return answer['result']
"""

"""
def ask_and_get_answer(vector_store, q, k=15):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain

    # Define the role for the LLM
    role = "L'assistant doit uniquement utiliser les informations contenues dans le vector_store pour répondre à la question. Il ne doit pas utiliser ses propres connaissances ou effectuer des recherches sur le web."

    # Create a prompt template with the role
    prompt_template = PromptTemplate(
        template=role + "\n\nContexte: {context}\nQuestion: {question}\nRéponse:",
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Create the stuff documents chain manually
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Create the retrieval QA chain
    chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain
    )
    
    # Ensure the correct keys are used when invoking the chain
    answer = chain({"query": q})
    return answer['result']"""




"""# Asking Questions and Getting Answers

def ask_and_get_answer(vector_store, q, k=15):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer['result']"""






# clear the chat history from streamlit session state

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

    