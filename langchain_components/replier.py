import os
import cohere
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Устанавливаем переменные окружения для ключа API OpenAI и Cohere
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

co = cohere.Client(cohere_api_key)

# Функция для создания истории сообщений в PostgreSQL
def create_postgres_chat_message_history(session_id, user_id):
    return PostgresChatMessageHistory(connection_string=POSTGRES_URL, session_id=session_id)

# Функция для подготовки шаблона подсказок и создания цепочки с историей сообщений
def prepare_prompt_and_chain_with_history():
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Вы – интеллектуальный помощник. Пожалуйста, обобщите содержание базы знаний для ответа на вопрос. Перечислите данные из базы знаний и ответьте подробно. Если все данные базы знаний не имеют отношения к вопросу, ваш ответ должен включать фразу "Ответ, который вы ищете, не найден в базе знаний!". Ответы должны учитывать историю чата.""",
            ),
            "Вот база знаний: {data}. Это база знаний.",
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
        ]
    )
    runnable = prompt | llm
    with_message_history = RunnableWithMessageHistory(
            runnable,
            create_postgres_chat_message_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="ID пользователя",
                    description="Уникальный идентификатор пользователя.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="ID сессии",
                    description="Уникальный идентификатор для разговора.",
                    default="",
                    is_shared=True,
                ),
            ],
            verbose=True,
        )
    return with_message_history

# Функция для получения хранилища векторных данных из PostgreSQL
def get_vectorstore_from_postgres(collection_name):
    openai_ef = OpenAIEmbeddings()
    vectorstore = PGVector(
        embeddings=openai_ef,
        collection_name=collection_name,
        connection=POSTGRES_URL,
        use_jsonb=True,
    ) 
    return vectorstore

# Функция ранжирования с использованием Cohere
def cohere_re_ranker(query, docs):
    # rerank_docs = co.rerank(query=query, documents=docs, top_n=4, model='rerank-english-v3.0')
    rerank_docs = co.rerank(query=query, documents=docs, top_n=4, model='rerank-multilingual-v3.0')
    
    return rerank_docs.results

# Функция для получения контекста из векторного хранилища
def get_context_from_vectorstore(user_query, vectorstore):
    relevant_docs = vectorstore.similarity_search(user_query, k=20)
    docs = [doc.page_content for doc in relevant_docs]
    rerank_docs = cohere_re_ranker(user_query, docs)
    result = ""
    reranked_index = []
    for doc in rerank_docs:
        index = doc.index
        reranked_index.append(index)
        result += docs[index] + "\n"
    logger.info(reranked_index)
    return result

