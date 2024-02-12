from flask import Flask ,render_template, request, redirect, url_for
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain import embeddings
from langchain.prompts import PromptTemplate
import datetime
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers import KNNRetriever


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
client = OpenAI()

llm=OpenAI(max_tokens=1024)
embeddings = embeddings.OpenAIEmbeddings()
# embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0.3)
persist_directory='db'


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        starttime = datetime.datetime.now()
        pdf_docs = request.files['fileInput']
        folder = 'pdfs'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder,pdf_docs.filename)
        pdf_docs.save(filepath)

        s2 = datetime.datetime.now()
        loader = PyPDFDirectoryLoader("pdfs")
        # data = loader.load_and_split()
        global data
        data = loader.load()
        context = "\n".join(str(p.page_content) for p in data)
        print("The total number of words in the context:", len(context))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        global texts    
        texts = text_splitter.split_text(context)

        # vectordb = Chroma.from_documents(documents=data,embedding=embeddings,persist_directory=persist_directory)
        # # persiste the db to disk
        # vectordb.persist()
        # vectordb = None

        vectordb = Chroma.from_texts(texts,embedding=embeddings,persist_directory=persist_directory)
        # persiste the db to disk
        vectordb.persist()
        vectordb = None

        endtime = datetime.datetime.now()
        totaltime = endtime- starttime
        print("total time for home : ",totaltime)
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        s1 = datetime.datetime.now()
        question = request.form['questionInput']

        # vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_type="mmr", search_kwargs={"k":1})
        # docs = vector_index.get_relevant_documents(question)

        vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
        print(vectordb)
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":1})
        docs = retriever.get_relevant_documents(question)

        # KNN = KNNRetriever.from_texts(texts, embeddings)
        # docs = KNN.get_relevant_documents(question,n_results=4,max_tokens=1024)

        # print(docs)

        s2 = datetime.datetime.now()
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details try your best to find details as powerfull ai system, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

        # model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,temperature=1)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        s3 = datetime.datetime.now()
        response = chain(
            {"input_documents":docs, "question": question}
            , return_only_outputs=True)
        # print(response)

        end = datetime.datetime.now()

        s1_time = s2 - s1
        s2_time = s3 - s2 
        s3 = end - s3

        print("s1 time : ", s1_time)
        print("s2_time : ", s2_time)
        print("s3_time : ", s3)
        return render_template("result.html", final_result=response['output_text'])

    return render_template('result.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)        


