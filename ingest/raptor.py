import os
import umap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


RANDOM_SEED = 224  # значение для генератора случайных чисел
openai_api_key = os.getenv("OPENAI_API_KEY")  

embd = OpenAIEmbeddings()  # Инициализация объекта для создания эмбеддингов с помощью OpenAI
model = ChatOpenAI(temperature=0, model="gpt-4o") 


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Выполняет глобальное сокращение размерности эмбеддингов с использованием UMAP.
    
    Параметры:
    - embeddings: Эмбеддинги, представленные в виде numpy массива.
    - dim: Целевая размерность для сокращения.
    - n_neighbors: Необязательный параметр; количество соседей для каждого элемента.
                   Если не задан, используется корень из количества эмбеддингов.
    - metric: Метрика для вычисления расстояний в UMAP.
    
    Возвращает:
    - numpy массив с эмбеддингами, уменьшенными до заданной размерности.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)  # Если количество соседей не задано, используем корень из числа эмбеддингов
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Выполняет локальное сокращение размерности эмбеддингов с использованием UMAP, обычно после глобальной кластеризации.
    
    Параметры:
    - embeddings: Эмбеддинги для кластеризации.
    - dim: Целевая размерность для сокращения.
    - num_neighbors: Количество соседей для каждого элемента.
    - metric: Метрика для вычисления расстояний.
    
    Возвращает:
    - numpy массив с локально сокращёнными эмбеддингами.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Определяет оптимальное количество кластеров с использованием Байесовского информационного критерия (BIC) и модели смешивания Гаусса.
    
    Параметры:
    - embeddings: Эмбеддинги для кластеризации.
    - max_clusters: Максимальное количество кластеров для поиска.
    - random_state: Параметр для воспроизводимости результатов.
    
    Возвращает:
    - Оптимальное количество кластеров.
    """
    max_clusters = min(max_clusters, len(embeddings))  # Ограничиваем максимальное количество кластеров
    n_clusters = np.arange(1, max_clusters)
    bics = []  # Список для хранения значений BIC для разных количеств кластеров
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)  # Создаем модель смешивания Гаусса
        gm.fit(embeddings)  # Обучаем модель
        bics.append(gm.bic(embeddings))  # Добавляем BIC для текущего количества кластеров
    return n_clusters[np.argmin(bics)]  # Возвращаем количество кластеров с минимальным BIC


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Кластеризует эмбеддинги с использованием модели смешивания Гаусса (GMM) на основе порога вероятности.
    
    Параметры:
    - embeddings: Эмбеддинги для кластеризации.
    - threshold: Порог вероятности для назначения эмбеддинга в кластер.
    - random_state: Параметр для воспроизводимости.
    
    Возвращает:
    - Массив меток кластеров и количество кластеров.
    """
    n_clusters = get_optimal_clusters(embeddings)  # Определяем оптимальное количество кластеров
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)  # Вычисляем вероятности для каждого эмбеддинга
    labels = [np.where(prob > threshold)[0] for prob in probs]  # Присваиваем метки на основе порога
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Выполняет кластеризацию эмбеддингов: сначала глобальное сокращение размерности, затем кластеризацию с использованием модели Гаусса,
    и, наконец, локальную кластеризацию внутри каждого глобального кластера.
    
    Параметры:
    - embeddings: Эмбеддинги для кластеризации.
    - dim: Целевая размерность для сокращения.
    - threshold: Порог для назначения эмбеддингов в кластер.
    
    Возвращает:
    - Список массивов меток кластеров для каждого эмбеддинга.
    """
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]  # Если слишком мало данных, возвращаем один кластер для всех
    
    # Глобальное сокращение размерности
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Глобальная кластеризация
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]  # Список для хранения локальных кластеров
    total_clusters = 0

    # Локальная кластеризация для каждого глобального кластера
    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Присваиваем метки локальных кластеров
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts):
    """
    Генерирует эмбеддинги для списка текстовых документов.

    Эта функция предполагает наличие объекта `embd` с методом `embed_documents`, 
    который принимает список текстов и возвращает их эмбеддинги.

    Параметры:
    - texts: List[str], список текстовых документов для создания эмбеддингов.

    Возвращает:
    - numpy.ndarray: Массив эмбеддингов для заданных текстов.
    """
    text_embeddings = embd.embed_documents(texts)  # Генерация эмбеддингов для текстов
    text_embeddings_np = np.array(text_embeddings)  # Преобразование эмбеддингов в NumPy массив
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Генерирует эмбеддинги для списка текстов и выполняет кластеризацию, возвращая DataFrame с текстами,
    их эмбеддингами и метками кластеров.

    Эта функция сочетает в себе генерацию эмбеддингов и кластеризацию. Предполагается наличие функции 
    `perform_clustering`, которая выполняет кластеризацию на эмбеддингах.

    Параметры:
    - texts: List[str], список текстовых документов для обработки.

    Возвращает:
    - pandas.DataFrame: DataFrame, содержащий оригинальные тексты, их эмбеддинги и назначенные метки кластеров.
    """
    text_embeddings_np = embed(texts)  # Генерация эмбеддингов
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Выполнение кластеризации на эмбеддингах
    df = pd.DataFrame()  # Инициализация DataFrame для хранения результатов
    df["text"] = texts  # Сохранение оригинальных текстов
    df["embd"] = list(text_embeddings_np)  # Сохранение эмбеддингов как списка в DataFrame
    df["cluster"] = cluster_labels  # Сохранение меток кластеров
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Форматирует текстовые документы из DataFrame в одну строку.

    Параметры:
    - df: DataFrame, содержащий колонку 'text' с текстовыми документами для форматирования.

    Возвращает:
    - Строка, в которой все текстовые документы соединены с использованием специфичного разделителя.
    """
    unique_txt = df["text"].tolist()  # Получение списка уникальных текстов
    return "--- --- \n --- --- ".join(unique_txt)  # Форматирование текстов в строку с разделителем


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Генерирует эмбеддинги, выполняет кластеризацию и подытоживает список текстов.
    Сначала генерируются эмбеддинги для текстов, затем они кластеризуются по схожести,
    расширяются назначения кластеров для удобства обработки и затем производится суммаризация
    содержимого в каждом кластере.

    Параметры:
    - texts: Список текстовых документов для обработки.
    - level: Целочисленный параметр, который может определять глубину или подробность обработки.

    Возвращает:
    - Кортеж, содержащий два DataFrame:
      1. Первый DataFrame (`df_clusters`) включает оригинальные тексты, их эмбеддинги и назначения кластеров.
      2. Второй DataFrame (`df_summary`) содержит суммарные данные для каждого кластера, 
         указанный уровень детализации и идентификаторы кластеров.
    """

    # Генерация эмбеддингов и кластеризация текстов, результатом будет DataFrame с 'text', 'embd' и 'cluster' колонками
    df_clusters = embed_cluster_texts(texts)

    # Подготовка к расширению DataFrame для более удобной манипуляции с кластерами
    expanded_list = []

    # Расширение DataFrame, чтобы каждая строка представляла текст и его кластер для более удобной обработки
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Создание нового DataFrame из расширенного списка
    expanded_df = pd.DataFrame(expanded_list)

    # Получение уникальных идентификаторов кластеров для дальнейшей обработки
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Сгенерировано {len(all_clusters)} кластеров--")

    # Подготовка шаблона для суммаризации
    template = """
    Пожалуйста, подытожьте следующие абзацы. Будьте внимательны с числами, не выдумывайте ничего. Абзацы следующие:
      {context}
    Это содержание, которое необходимо подытожить.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    # Форматирование текста в каждом кластере для суммаризации
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Создание DataFrame для хранения суммарных данных с их соответствующими кластерами и уровнями
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Рекурсивно генерирует эмбеддинги, выполняет кластеризацию и суммаризацию текстов до заданного уровня или
    до тех пор, пока количество уникальных кластеров не станет 1, сохраняя результаты на каждом уровне.

    Параметры:
    - texts: List[str], тексты для обработки.
    - level: int, текущий уровень рекурсии (начинается с 1).
    - n_levels: int, максимальная глубина рекурсии.

    Возвращает:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], словарь, где ключи — это уровни рекурсии, 
      а значения — кортежи, содержащие DataFrame с кластерами и суммаризируемые DataFrame на каждом уровне.
    """
    results = {}  # Словарь для хранения результатов на каждом уровне

    # Выполнение эмбеддинга, кластеризации и суммаризации для текущего уровня
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # Сохранение результатов текущего уровня
    results[level] = (df_clusters, df_summary)

    # Определение, возможно ли продолжение рекурсии
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Использование суммарных данных как новых текстов для следующего уровня рекурсии
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Объединение результатов следующего уровня в текущий словарь результатов
        results.update(next_level_results)

    return results
