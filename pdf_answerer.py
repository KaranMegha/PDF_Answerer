# pip install langchain
# pip install openai
# pip install PyPDF2
# pip install faiss-cpu
# pip install tiktoken

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"

# location of the pdf file/files.
reader = PdfReader('./e-Notes_PDF_All-Units_29042019072534AM.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

docsearch

chain = load_qa_chain(OpenAI(), chain_type="stuff")

def prompt():
  print("<------------------------------------------New Question-------------------------------------->")
  query = str(input("Ask any question :"))
  docs = docsearch.similarity_search(query)
  answer = chain.run(input_documents=docs, question=query)
  print("<------------------------------------------Answer-------------------------------------->")
  print(answer)
  prompt()

# prompt()

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route("/",methods=["GET","POST"])
@cross_origin()
def result():
   requestParam = request.get_json()
   query = str(requestParam['question'])
   docs = docsearch.similarity_search(query)
   answer = chain.run(input_documents=docs, question=query)
   return {"Answer": answer}

if __name__ == '__main__':
   app.run(debug=True,port=2000)