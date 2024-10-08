
#import faiss
import concurrent.futures
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.llms import AzureOpenAI
import numpy as np
import re
import logging
import pandas as pd
import pandas as pd
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import pandas as pd
from langchain.prompts import PromptTemplate
import os
import openai
import psycopg2
import psycopg2.extras


# initialize LLM and Embeddings LLM
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "your_api_base"
os.environ["OPENAI_API_KEY"] = "your_api_key"
llm = AzureOpenAI(engine="llm-acelerator", model_name="gpt-3.5-turbo", temperature=0)
MODEL = "text-embedding-ada-002"
embedding = OpenAIEmbeddings(
                deployment="llm-acelerator-embedding",
                model=MODEL,
                )


# Initialize OpenAI API with the key from environment variables

connection = psycopg2.connect(
    dbname="your_database_name",
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432"
)


articles = [
    "Python is a highly versatile and widely used programming language, known for its ease of learning and flexibility in application development...",
    "JavaScript stands as the backbone of modern web development, powering the dynamic behavior on the majority of websites...",
    "Version control systems (VCS) are fundamental tools in the realm of software development, providing teams with the ability to manage changes...",
    "Digital marketing has undergone a remarkable transformation over the past decade, adapting to changing consumer behaviors...",
    "Urban green spaces, such as parks, gardens, and river walkways, play a crucial role in enhancing the quality of life in cities..."
]

with connection.cursor() as cursor:
    for article in articles:
        response = openai.Embedding.create(
                input=article,
                model="text-embedding-ada-002"  # Adjust the model name if necessary
            )
        print("Started")
        print(embedding)
        print("Done")
        '''
        embedding_vector = embedding['data'][0]['embedding']
        print(embedding_vector)
        cursor.execute(
                'INSERT INTO articles (content, embedding) VALUES (%s, %s)',
                (article, psycopg2.extras.Json(embedding_vector))
            )
        '''
    # Commit the transaction
    connection.commit()



# Close the database connection
connection.close()
