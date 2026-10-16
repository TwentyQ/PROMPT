import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ast
from langchain_core.embeddings import Embeddings
import time
from typing import List


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    """
    Загружает данные эмбеддингов из CSV файла в pandas DataFrame

    Args:
        filepath (str): Путь к файлу с данными

    Returns:
        pd.DataFrame: DataFrame с загруженными данными
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {len(df)} записей из файла {filepath}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"\nПервые 3 записи:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None


# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")

# Просмотр информации о датасете
print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПроверка пропущенных значений:\n{df.isnull().sum()}")
print("Доступные столбцы:", df.columns.tolist())

print("\n" + "=" * 80)

# Статистика по категориям
print("\nСтатистика по категориям")

# Основные категории (топ-20)
print(f"\nОсновные категории")
main_cat_stats = df['main_category'].value_counts()
for i, item in enumerate(main_cat_stats.head(20).items(), 1):
    category, count = item  # распаковываем здесь
    percentage = (count / len(df)) * 100
    print(f"{i:2}. {category}: {count} статей ({percentage:.1f}%)")

# Количество категорий на статью
print(f"\nКоличество категорий на статью:")
df['num_categories'] = df['categories'].str.split(',').str.len().fillna(0)
category_counts = df['num_categories'].value_counts().sort_index()
for num_cats, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {num_cats} категория(й): {count} статей ({percentage:.1f}%)")

# Анализ всех категорий
print(f"\nАнализ всех категорий:")
all_categories = []
for categories_str in df['categories'].dropna():
    categories_list = str(categories_str).replace('{', ' ').replace('}', ' ').replace(',', ' ').split()
    all_categories.extend(categories_list)

# Создаем Series(умного списка) из всех категорий
all_categories_series = pd.Series(all_categories)

# Базовая статистика
print(f"  Всего упоминаний категорий: {len(all_categories)}")
print(f"  Уникальных категорий: {all_categories_series.nunique()}")

# Топ-20 самых частых категорий
top_categories = all_categories_series.value_counts().head(20)
print(f"\n  Топ-20 самых частых категорий:")
for i, (cat, count) in enumerate(top_categories.items(), 1):
    percentage = (count / len(all_categories)) * 100
    print(f"  {i:2}. {cat}: {count} упоминаний ({percentage:.1f}%)")

print("\n" + "=" * 80)

# Случайный выбор подмножества из 1000 статей
df_min = df.sample(n=1000, random_state=42, replace=False)


def prepare_documents(df_min, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain

    Args:
        df_min: DataFrame с данными статей
        limit: Количество документов для обработки

    Returns:
        List[Document]: Список документов
    """
    documents = []

    df_subset = df_min.head(limit)

    for idx, row in df_subset.iterrows():
        # Формирование содержимого документа
        page_content = ""

        if 'title' in row:
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in row:
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in row:
            page_content += f"Авторы: {row['authors']}\n"

        # Метаданные
        metadata = {}
        if 'main_category' in row and pd.notna(row['main_category']):
            metadata['main_category'] = row['main_category']
        if 'categories' in row and pd.notna(row['categories']):
            metadata['categories'] = row['categories']
        if 'year' in row and pd.notna(row['year']):
            metadata['year'] = int(row['year'])
        if 'article_id' in row and pd.notna(row['article_id']):
            metadata['article_id'] = row['article_id']

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    print(f"Подготовлено {len(documents)} документов")
    return documents


# Подготовка документов
documents = prepare_documents(df_min, limit=1000)

# Просмотр примера документа
print("Пример документа:")
print(f"Содержимое: {documents[0].page_content[:200]}...")
print(f"Метаданные: {documents[0].metadata}")

print("\n" + "=" * 80)

# Конфигурация OpenAI клиента для получения эмбеддингов
client = OpenAI(
    api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


# Функция для получения эмбеддингов
def get_embedding(text: str, model="BAAI/bge-m3") -> list:
    """Получает эмбеддинг текста"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


# Тестирование функции эмбеддингов
test_embedding = get_embedding("Тестовый запрос")
print(f"Размерность эмбеддинга: {len(test_embedding)}")


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)

# Создание векторного хранилища ChromaDB
# print("Создание векторного хранилища...")
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     collection_name="arxiv_papers",
#     persist_directory="./chroma_db"  # Директория для сохранения
# )
# print("Векторное хранилище успешно создано!")

# Загрузка существующего векторного хранилища ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)
print("Векторное хранилище загружено!")

# # Проверка работы хранилища
# test_query = "модели для генерации"
# results = vectorstore.similarity_search(test_query, k=3)
# print(f"\nРезультаты поиска по запросу '{test_query}':")
# for i, doc in enumerate(results, 1):
#     print(f"\n{i}. {doc.page_content[:200]}...")
#     print(f"   Метаданные: {doc.metadata}")


'''
Эксперименты со значениями k и различными запросами
'''
# print("Разные значения параметра k (количество документов):")
# query = 'модели для генерации'
# for k in [1, 3, 5, 10]:
#     # Используем метод similarity_search_with_score для получения оценок
#     results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
#
#     print(f"\nk = {k}:")
#     print(f"Найдено документов: {len(results_with_scores)}")
#
#     for i, (doc, score) in enumerate(results_with_scores, 1):
#         # Получаем название из первой строки содержимого
#         print(f"  {i}. Оценка: {score:.4f} - {doc.page_content[:200]}...")
#
#
#
# print("\nПоиск по разным тематическим запросам (k=3):")
# queries = [
#     "нейронные сети и искусственный интеллект",
#     "обработка естественного языка",
#     "компьютерное зрение и распознавание изображений",
#     "трансформеры",
#     "регрессия в машинном обучении"
# ]
#
# for query_idx, query in enumerate(queries, 1):
#     print(f"{query_idx}. Запрос: '{query}'")
#     results = vectorstore.similarity_search_with_score(query, k=3)
#     for i, (doc, score) in enumerate(results, 1):
#         # Получаем название из первой строки содержимого
#         print(f"  {i}. Оценка: {score:.4f} - Категория: {category} -  {doc.page_content[:50]}...")


print("\n" + "=" * 80)

# Создание ретривера с поиском по схожести(базового)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Возвращать топ-5 документов
)
# Тестирование ретривера
# query = "глубокое обучение для обработки изображений"
# retrieved_docs = retriever.invoke(query)
# print(f"Найдено документов: {len(retrieved_docs)}")
# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"\nДокумент {i}:")
#     print(doc.page_content[:150] + "...")


# MMR ретривер
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Количество документов для первичной выборки
        "lambda_mult": 0.5  # Баланс между релевантностью (1.0) и разнообразием (0.0)
    }
)

# Ретривер с фильтрацией по оценке схожести
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Минимальная оценка схожести
        "k": 10
    }
)

# Сравнение результатов
print("\nСравнение результатов поиска:")

query = "обработка естественного языка"
docs_similarity = retriever.invoke(query)
docs_mmr = retriever_mmr.invoke(query)

print("\n=== Similarity Search ===")
for i, doc in enumerate(docs_similarity[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n=== MMR Search ===")
for i, doc in enumerate(docs_mmr[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n" + "=" * 80)

print("Эксперименты с параметром lambda_mult в MMR:")

query_mmr = "искусственный интеллект"

# Тестируем разные значения lambda_mult
lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]

for lambda_val in lambda_values:
    print(f"\nlambda_mult = {lambda_val}:")

    retriever_mmr_test = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 100,
            "lambda_mult": lambda_val
        }
    )

    # Получаем документы
    docs_mmr_test = retriever_mmr_test.invoke(query_mmr)

    print(f"  Найдено документов: {len(docs_mmr_test)}")

    # Анализ разнообразия категорий
    categories = []
    for i, doc in enumerate(docs_mmr_test, 1):
        category = doc.metadata.get('main_category')
        categories.append(category)
        print(f"  {i}. - Категория: {category} - {doc.page_content[:100]}...")

    unique_cats = set(categories)
    diversity_ratio = len(unique_cats) / len(categories) if categories else 0
    print(f"{len(unique_cats)} уникальных категорий из {len(categories)}")
    print(f"Коэффициент разнообразия: {diversity_ratio:.2f}")

print("\nСравнение времени работы разных типов поиска:")


def search_time(retriever, query, iterations=3):
    """Измеряет среднее время поиска"""
    times = []

    for _ in range(iterations):
        start_time = time.time()
        results = retriever.invoke(query)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    return avg_time, len(results) if 'results' in locals() else 0


# Создание разных ретриверов для сравнения
retrievers = {
    "similarity (k=3)": vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    ),
    "similarity (k=10)": vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    ),
    "mmr (lambda=0.0)": vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.0}
    ),
    "mmr (lambda=0.5)": vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
    ),
    "mmr (lambda=1.0)": vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 1.0}
    ),
    "similarity_threshold (0.3)": vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )
}

# Тестовые запросы
test_queries = [
    "глубокое обучение",
    "агенты",
    "компьютерное зрение"
]

for query in test_queries:
    print("-" * 80)
    print(f"\nЗапрос: '{query}'")

    results_table = []

    for retriever_name, retriever in retrievers.items():
        avg_time, docs_count = search_time(retriever, query, iterations=3)
        results_table.append({
            'name': retriever_name,
            'time': avg_time,
            'docs': docs_count
        })

    # Сортировка по времени
    results_table.sort(key=lambda x: x['time'])

    # Таблица результатов
    print(f"{'Тип поиска':<30} {'Время (сек)':<30} {'Документов':<30}")
    print("-" * 80)
    for result in results_table:
        print(f"{result['name']:<30} {result['time']:<30} {result['docs']:<30}")

    # Анализ
    fastest = results_table[0]
    slowest = results_table[-1]
    speedup = slowest['time'] / fastest['time']

    print(f"\n  Самый быстрый: {fastest['name']} ({fastest['time']:.4f} сек)")
    print(f"  Самый медленный: {slowest['name']} ({slowest['time']:.4f} сек)")
    print(f"  Разница: {speedup:.1f}")

print("\n" + "=" * 80)

# Настройка языковой модели
# Пример конфигурации
llm = GigaChat(
    credentials="OThhZGViNTgtN2E0Mi00YmExLTgzMTctM2YwNjFmNGI0NzNkOmM2YzYzMGJlLTczMGQtNDk3MC04MjRlLWQwZjBkZWRkM2U5Mg==",
    scope="GIGACHAT_API_B2B",
    model="Gigachat-2-Pro",
    verify_ssl_certs=False,
    timeout=30
)

# Тест языковой модели
# test_response = llm.invoke("Привет! Ответь кратко: что такое машинное обучение?")
# print(f"Ответ модели: {test_response.content}")


# Шаблон промпта для RAG
prompt_template = """Ты -- научный ассистент, специализирующийся на анализе научных статей.
Твоя задача -- отвечать на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте из научных статей ArXiv.

Правила:
1. Используй только информацию из контекста ниже
2. Если в контексте нет информации для ответа, честно скажи об этом
3. Указывай, из каких статей взята информация (если есть метаданные)
4. Отвечай на русском языке, четко и структурированно
5. Если вопрос касается технических деталей, будь точным

Контекст из научных статей:
{context}

Вопрос пользователя: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)


def format_docs(docs):
    """
    Форматирует список документов в единую строку контекста

 Args:
        docs: Список документов Document

    Returns:
        str: Форматированный контекст
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Документ {i}]")
        context_parts.append(doc.page_content)
        if doc.metadata:
            context_parts.append(f"Метаданные: {doc.metadata}")
        context_parts.append("")  # Пустая строка для разделения
    return "\n".join(context_parts)


# Тест форматирования
# test_docs = retriever.invoke("нейронные сети")
# formatted_context = format_docs(test_docs[:2])
# print("Пример форматированного контекста:")
# print(formatted_context[:500] + "...")

# Создание RAG-цепочки
rag_chain = (
        {
            "context": retriever | format_docs,  # Извлекаем и форматируем документы
            "question": RunnablePassthrough()  # Передаем вопрос как есть
        }
        | prompt  # Формируем промпт
        | llm  # Отправляем в языковую модель
)

# Тестирование RAG-системы
print("=== Тестирование RAG-системы ===\n")

questions = [
    "Как машинное обучение помогает в медицине?",
    "Расскажи о применении агентов",
    "Как используются нейронные сети?"
]

for i, question in enumerate(questions, 1):
    print(f"Вопрос {i}: {question}")
    print("-" * 80)

    try:
        response = rag_chain.invoke(question)
        print(f"Ответ: {response.content}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")


def interactive_rag_qa():
    """Интерактивная система вопросов-ответов"""
    print("=== Интерактивная RAG-система для научных статей ArXiv ===")
    print("Введите 'выход' для завершения\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not question:
            continue

        try:
            # Получаем релевантные документы
            docs = retriever.invoke(question)
            print(f"\nНайдено релевантных документов: {len(docs)}")

            # Генерируем ответ
            response = rag_chain.invoke(question)
            print(f"\nОтвет:\n{response.content}\n")

            # Показываем источники
            show_sources = input("Показать источники? (да/нет): ").strip().lower()
            if show_sources in ['да', 'yes', 'y', 'д']:
                print("\nИсточники:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    print(f"   Метаданные: {doc.metadata}")

            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"Ошибка: {e}\n")

# Запуск интерактивной системы (раскомментируйте для использования)
# interactive_rag_qa()
