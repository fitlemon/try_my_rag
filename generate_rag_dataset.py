#!/usr/bin/env python
# coding: utf-8

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import re
from typing import List

# Load environment variables
load_dotenv()

# ### Load the document
loader = PyPDFLoader("data/labor_codex.pdf")
docs = loader.load()

# Combine all document text
text = "".join(doc.page_content for doc in docs)


from langchain.schema import Document
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
import re
from typing import List


class RegexTextSplitter(TextSplitter):
    def __init__(
        self,
        chapter_pattern: str,
        section_pattern: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """
        :param chapter_pattern: Regex pattern for chapters (e.g., r"(\\d+-bob)")
        :param section_pattern: Regex pattern for articles (modda), (e.g., r"(\\d+-modda)")
        :param chunk_size: Max tokens per chunk
        :param chunk_overlap: Overlap tokens between consecutive chunks
        """
        self.chapter_pattern = chapter_pattern
        self.section_pattern = section_pattern
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Splitting text into chunks using GPT-4 tokenizer (can be replaced with another model)
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_by_chapter(self, text: str) -> List[dict]:
        """
        Splits the entire text by chapters according to chapter_pattern.
        Returns a list of dicts like:
        [
        {"chapter_title": "II BO‘LIM. MEHNAT SOHASIDAGI IJTIMOIY SHERIKLIK", "content": "Text of that chapter"},
        ...
        ]
        """
        chapters = []
        pattern = (
            rf"({self.chapter_pattern})\s*"  # group(1): chapter number/title, e.g., "II BO‘LIM."
            rf"([\s\S]*?)(?=\n[0-9])"  # group(2): chapter name lazily until \n followed by digit
            rf"\n([\s\S]*?)(?={self.chapter_pattern}|$)"  # group(3): chapter content until next chapter or EOF
        )

        chapter_matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in chapter_matches:
            chapter_title = (
                f"{match[0].strip()} {match[1].strip()}"  # Combine number and title
            )
            chapter_content = match[2].strip()  # Extract content
            chapters.append(
                {"chapter_title": chapter_title, "content": chapter_content}
            )
        return chapters

    def split_by_section(self, chapter: dict) -> List[dict]:
        """
        We assume each article (modda) can look like:

            1-modda. Article Title
            The rest of the article content...

        but sometimes the "Article Title" might be on the same line or might
        not have a trailing newline.

        This pattern:
        - Group(1): e.g. "1-modda"
        - Group(2): the article title (from after "." up to the newline or end of line)
        - Group(3): (optional) the rest of that article's text (until the next modda, bob, or EOF)
        """

        text = chapter["content"]

        pattern = (
            rf"({self.section_pattern})\.\s*"  # group(1): modda number, e.g. "578-modda"
            rf"([\s\S]*?)(?=\n[A-ZА-Я])"  # group(2): captures text lazily until we see newline + uppercase
            rf"(.*?)(?=\n{self.section_pattern}|$)"
        )

        matches = re.findall(pattern, text, flags=re.DOTALL)

        sections = []
        for match in matches:
            section_title = match[1].strip()  # e.g. "1-modda"
            section_name = match[2].strip()  # e.g. "Ushbu Kodeks bilan..."
            section_body = (
                match[3].strip() if match[3] else ""
            )  # rest of the modda text

            sections.append(
                {
                    "chapter_title": chapter["chapter_title"],
                    "section_title": section_title,
                    "section_name": section_name.replace("\n", ""),
                    "content": section_body,
                }
            )
        # print(matches)
        return sections

    def split_with_chunks(self, section: dict) -> List[Document]:
        """
        Breaks the content of a single article (section) into token-based chunks
        using RecursiveCharacterTextSplitter.
        """
        chunks = self.recursive_splitter.split_text(section["content"])
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter": section["chapter_title"],
                    "section": section["section_title"],
                    "section_name": section["section_name"],
                },
            )
            documents.append(doc)
        return documents

    def split_text(self, text: str) -> List[Document]:
        """
        1) Split by chapters -> 2) for each chapter, find all modda -> 3) chunk each modda
        """
        all_documents = []
        chapters = self.split_by_chapter(text)
        for chapter in chapters:
            sections = self.split_by_section(chapter)
            for section in sections:
                chunks = self.split_with_chunks(section)
                all_documents.extend(chunks)
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Runs the splitting process for each document in the list.
        """
        split_documents = []
        for doc in documents:
            split_docs = self.split_text(doc.page_content)
            split_documents.extend(split_docs)
        return split_documents


# ### Splitting Documents
chapter_pattern = r"(?:[IVXLCDM]+ BO‘LIM\.)"
section_pattern = r"(\d+-modda)"
splitter = RegexTextSplitter(
    chapter_pattern, section_pattern, chunk_size=800, chunk_overlap=400
)
doc = Document(page_content=text)
docs = splitter.split_text(doc.page_content)

# Metadata correction
for doc in docs:
    doc.metadata["chapter"] = doc.metadata["chapter"].replace(
        "31-bob. Umumiy qoidalar", ""
    )

# ### Labeling with LLM
label_template = """
You are an AI assistant tasked with generating 10 highly focused question-answer pairs **in Uzbek** based **strictly** on the provided document chunk.

Context:
- Document: "{chunk}"
- Chapter Name: "{chapter}"
- Section: "{section}"
- Section Name: {section_name}"

Instructions:
1. Carefully analyze the given chunk of the document and extract key facts, topics, or concepts.
2. Generate **10 questions** that are maximally semantically close to the specific content of this chunk.
    - All questions must be **strictly tied** to the content of the chunk, chapter, and section.
    - Reuse or closely paraphrase phrases from the chunk wherever possible to maintain high relevance.
    - Incorporate synonyms or light rephrasings to capture different ways a user might query this content (e.g., "mehnat huquqlari" ↔ "ish huquqlari").
    - Avoid any extrapolation, assumptions, or references to external knowledge.
3. Write the **answers** based only on the given chunk. **Do not invent or assume additional information.**
4. Explicitly reference the **chapter name** and **section name** in both the questions and answers for clarity and traceability.
5. Use natural, conversational Uzbek language when generating questions. Occasionally include mild typos or colloquialisms to mimic real user queries, but ensure they remain understandable.
6. Include a reference link in each answer pointing to the document, chapter, and section for traceability.
7. All questions must revolve around a **common theme** in the chunk (e.g., “mehnat huquqlari,” “ish beruvchilarning huquqlari,” etc.).

**Example of how to use phrases from the chunk (paraphrasing)**:
- If the chunk states: "Mehnat qilish, erkin ish tanlash va ishsizlikdan himoyalanish huquqi davlat kafolatlari bilan belgilanadi."
  - A suitable question might be: "Mehnat qilish va erkin ish tanlash huquqlari haqida {{chapter}} va {{section}} nima deyilgan?"
  - The answer must be grounded in the exact text: "Bu huquqlar davlat kafolatlari orqali belgilanadi. Havola: O‘ZBEKISTON RESPUBLIKASINING MEHNAT KODEKSI. {{chapter}}. {{section}}. {{section_name}}."

**Answer Example**:
If the document says: "Mehnat huquqlari himoyasi mehnat kodeksining asosiy maqsadi hisoblanadi."
- Question: "Mehnat kodeksining asosiy maqsadi nima?"
- Answer: "Mehnat huquqlarining himoyasi. Havola: O‘ZBEKISTON RESPUBLIKASINING MEHNAT KODEKSI. {{chapter}}, {{section}}. {{section_name}}."

Output Format:
Return a valid JSON object with the following structure:
```json
{{
  "question_1": "Generated question text",
  "answer_1": "Generated answer text",
  "question_2": "Generated question text",
  ...
}}
"""

label_prompt = ChatPromptTemplate.from_template(label_template)
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

label_chain = label_prompt | llm | JsonOutputParser()
label_prompt = ChatPromptTemplate.from_template(label_template)
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
label_chain = label_prompt | llm | JsonOutputParser()

# ### Generate Dataset
dataset = {}

print("Generating question-answer pairs:")
for i, doc in enumerate(tqdm(docs, desc="Processing chunks")):
    try:
        output = label_chain.invoke(
            {
                "chunk": doc.page_content,
                "chapter": doc.metadata["chapter"],
                "section": doc.metadata["section"],
                "section_name": doc.metadata["section_name"],
            }
        )
        dataset[f"doc_{i}"] = {
            "chunk": doc.page_content,
            "chapter": doc.metadata["chapter"],
            "section": doc.metadata["section"],
            "section_name": doc.metadata["section_name"],
            "questions": output,
        }
    except Exception as e:
        print(f"Error processing doc_{i}: {e}")
# save the dataset
try:
    import json

    with open("data/labor_codex_raw_dataset.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"Error saving dataset: {e}")
# ### Flatten and Save Dataset
dataset_rows = []
print("Flattening dataset for export:")
for doc_id, doc_data in tqdm(dataset.items(), desc="Flattening data"):
    chunk = doc_data["chunk"]
    chapter = doc_data["chapter"]
    section = doc_data["section"]
    section_name = doc_data["section_name"]
    questions = doc_data["questions"]

    for i in range(1, len(questions) // 2 + 1):
        dataset_rows.append(
            {
                "chunk": chunk,
                "chapter": chapter,
                "section": section,
                "section_name": section_name,
                "question": questions[f"question_{i}"],
                "answer": questions[f"answer_{i}"],
            }
        )

df = pd.DataFrame(dataset_rows)
df.to_pickle("data/labor_codex_rag_dataset.pkl")
df.to_json(
    "data/labor_codex_rag_dataset.json",
    orient="records",
    lines=True,
    force_ascii=False,
)
