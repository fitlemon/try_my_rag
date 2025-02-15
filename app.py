import os
import gradio as gr
import spaces
from datasets import load_dataset, concatenate_datasets
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
import pickle

# Импорты для перевода
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from uz_translit import to_latin

# Загружаем переменные окружения
load_dotenv()
hf_key = os.getenv("HF_KEY")

# Путь для сохранения FAISS-индекса
INDEX_PATH = "./faiss_index"

# Инициализируем эмбеддинг-модель
embeddings = HuggingFaceEmbeddings(model_name="fitlemon/bge-m3-uz-legal-matryoshka")
translations = pickle.load(open("translations.pkl", "rb"))


def update_faiss_index():
    """
    Загружает датасеты, преобразует данные в документы с метаданными,
    создаёт FAISS-индекс и сохраняет его локально.
    """
    train_dataset = load_dataset("fitlemon/rag-labor-codex-dataset", token=hf_key)[
        "train"
    ]
    test_dataset = load_dataset("fitlemon/rag-labor-codex-dataset", token=hf_key)[
        "test"
    ]

    dataset = concatenate_datasets([train_dataset, test_dataset])
    # dataset = dataset.select(range(5))  # Для тестирования на небольшом количестве данных
    docs = []
    unique_chunks = set()
    for row in tqdm(dataset, desc="Загрузка документов..."):
        chunk = row["chunk"]
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
    db = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    db.save_local(INDEX_PATH)
    print("FAISS индекс обновлён и сохранён в:", INDEX_PATH)
    return db


if not os.path.exists(INDEX_PATH):
    db = update_faiss_index()
else:
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Загружен существующий FAISS индекс из:", INDEX_PATH)


def translate_ru_uz(message: str) -> str:
    """
    Переводит текст с русского на узбекский с использованием ChatOpenAI.
    Пример: input: "отпуск" → output: "tatil".
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}. The subject of Text is Human Resources. Example input: отпуск. Output: tatil.",
            ),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini-2024-07-18"
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "input_language": "Russian",
            "output_language": "Uzbek",
            "input": message,
        }
    )
    return response.content


@spaces.GPU
def retrieve_articles(query, language):
    """
    Если выбран язык "Russian", переводит запрос с русского на узбекский.
    Затем ищет в FAISS-индексе топ-3 наиболее релевантных документа и возвращает результат в Markdown.
    """
    if language == "Russian":
        translated_query = translate_ru_uz(query)
    else:
        translated_query = to_latin(query)

    results = db.similarity_search(translated_query, k=3)

    result_text = ""
    for doc in results:
        result_text += (
            f"### {doc.metadata['section']}: {doc.metadata['section_name']}\n"
        )
        if language == "Russian":
            result_text += f"**Текст статьи на русском:** {translations.get(doc.page_content, 'Не найден')}\n\n"
        result_text += f"**Bo'lim:** {doc.metadata['chapter_name']}\n\n"
        result_text += f"**Modda teksti:**\n{doc.page_content}\n\n"
        result_text += "---\n\n"
    return result_text
    # return "Привет, мир!" if language == "Russian" else "Salom Dunyo!"


def toggle_language(current_language: str) -> gr.update:
    """
    Переключает язык между "Russian" и "Uzbek".
    """
    new_language = "Uzbek" if current_language == "Russian" else "Russian"
    return gr.update(value=new_language)


# Создаём Gradio-интерфейс на основе Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Поиск по Кодексу через Эмбеддинг Модель")
    gr.Markdown(
        "Введите ваш вопрос и выберите язык запроса. Если выбран русский, запрос будет переведен на узбекский перед поиском."
    )

    with gr.Row():
        language_radio = gr.Radio(
            choices=["Russian", "Uzbek"], label="Язык запроса", value="Russian"
        )

    query_input = gr.Textbox(
        lines=3, placeholder="Введите ваш вопрос о кодексе...", label="Запрос"
    )
    search_button = gr.Button("Поиск")

    output_markdown = gr.Markdown()

    search_button.click(
        fn=retrieve_articles,
        inputs=[query_input, language_radio],
        outputs=output_markdown,
    )
    query_input.submit(
        fn=retrieve_articles,
        inputs=[query_input, language_radio],
        outputs=output_markdown,
    )

if __name__ == "__main__":
    demo.launch()
