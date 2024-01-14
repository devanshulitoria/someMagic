from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en",
                                               model_kwargs={"device": 'cuda'})
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":6})
template="""you are a bot that helps people plan there travel itinerary by using only the context provided below. if you don't know the answer simple say that you don't know. Do not make up answers.
    
    {context}
    
    Question: {question}"""
prompt= PromptTemplate(input_variables=["context", "question"],template=template)
llm=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt":prompt,}, return_source_documents=True )
while True:
    question = input("Enter your question ")
    if question == "exit":
        break
    res = qa(question)
    answer, docs = res['result'], res['source_documents']
    print("\n\n> Question:",question)
    print("\n> Answer:",answer)
    print("\n\n\n documents are ",docs)
