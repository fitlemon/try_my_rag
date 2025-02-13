import os
import gradio as gr
from datasets import load_dataset, concatenate_datasets
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# Путь для сохранения FAISS-индекса
INDEX_PATH = "./faiss_index"

# Инициализируем эмбеддинг-модель (используем модель из Hugging Face)
embeddings = HuggingFaceEmbeddings(model_name="fitlemon/bge-m3-uz-legal-matryoshka")


def update_faiss_index():
    """
    Загружает датасет, преобразует данные в документы с метаданными,
    создаёт FAISS-индекс и сохраняет его локально.
    """
    # Загружаем датасет (например, сплит "train")
    train_dataset = load_dataset("fitlemon/rag-labor-codex-dataset")["train"]
    test_dataset = load_dataset("fitlemon/rag-labor-codex-dataset")["test"]
    # combine train and test datasets
    dataset = concatenate_datasets([train_dataset, test_dataset])
    # get rid off duplicate chunks

    docs = []
    unique_chunks = set()
    for row in tqdm(dataset, desc="Загрузка документов..."):
        chunk = row["chunk"]
        # Если chunk уже добавлен, пропускаем его
        if chunk in unique_chunks:
            continue
        unique_chunks.add(chunk)

        doc = Document(
            page_content=chunk,
            metadata={
                "section": row["section"],
                "section_name": row["section_name"],
                "chapter_name": row["chapter"],
            },
        )
        docs.append(doc)

    print(f"Документы успешно загружены и преобразованы. Длина документов: {len(docs)}")
    # Создаём FAISS-индекс на основе документов
    db = FAISS.from_documents(docs, embeddings)

    # Сохраняем индекс в указанную директорию
    os.makedirs(INDEX_PATH, exist_ok=True)
    db.save_local(INDEX_PATH)
    print("FAISS индекс обновлён и сохранён в:", INDEX_PATH)
    return db


# Если индекс ещё не создан, обновляем его, иначе загружаем существующий
if not os.path.exists(INDEX_PATH):
    db = update_faiss_index()
else:
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Загружен существующий FAISS индекс из:", INDEX_PATH)


def retrieve_articles(query):
    """
    Принимает запрос пользователя, ищет в FAISS-индексе топ-3 наиболее релевантных документа
    и возвращает отформатированный результат в Markdown.
    """
    # Поиск по индексу: возвращает список из документов
    results = db.similarity_search(query, k=3)

    # Форматируем результаты для вывода
    result_text = ""
    for doc in results:
        result_text += (
            f"### Статья {doc.metadata['section']}: {doc.metadata['section_name']}\n"
        )
        result_text += f"**Глава:** {doc.metadata['chapter_name']}\n\n"
        result_text += f"**Текст статьи:**\n{doc.page_content}\n\n"
        result_text += "---\n\n"
    return result_text


# Создаём Gradio-интерфейс
iface = gr.Interface(
    fn=retrieve_articles,
    inputs=gr.Textbox(lines=3, placeholder="Введите ваш вопрос о кодексе..."),
    outputs=gr.Markdown(),
    title="Поиск по Кодексу через FAISS",
    description="Введите вопрос, и получите топ-3 наиболее релевантные статьи из кодекса.",
)

if __name__ == "__main__":
    iface.launch()
