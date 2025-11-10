# 📚 Quran Chatbot: Gen Z Vibe (Code: ON) 🚀

This project is an AI chatbot that answers questions about the Quran's meaning (Meal) and commentary (Tafsir) using a young, friendly, and **chill** language (Gen Z slang and lots of emojis). The **vibe is high**, but the language is always **respectful and sensitive** towards the holy texts and values.

---

## 💡 Quick Technical Overview

- **Large Language Model (LLM):** `gemini-2.5-flash` (Focused on speed and efficiency)
- **Embedding Model:** `BGE-M3 (Turkish)` (High performance for Turkish texts)
- **Vector Database:** Chroma DB
- **Interface Framework:** Gradio

---

## ✨ Main Features (Why is it so cool?)

- **Gen Z Tone:** Answers are **friendly, witty, full of slang**, and include lots of emojis. It avoids philosophical or overly formal language.
- **Respect Control:** **Absolute sensitivity** towards holy texts and religious concepts (This doesn't lower the vibe; it makes it **cool**).
- **RAG (Retrieval-Augmented Generation):** Instead of making up information, it answers by pulling evidence from the actual Quran meaning and commentary texts.
- **Sectional Reading:** When you enter a Surah name, it shares long Surahs in **parts** to create the right reading **mood**.

---

## 🗣️ Gen Z Tone and Command Chain (SYSTEM_INSTRUCTION)

To keep the chatbot's "chill" and friendly character, a very detailed **SYSTEM_INSTRUCTION** text is sent to the Large Language Model (LLM).  
This instruction defines the rules the model must follow in every answer and explains the RAG chain's operation:

| **Component** | **Role** | **Description** |
|----------------|-----------|-----------------|
| **SYSTEM_INSTRUCTION** | **Persona and Rules** | Sets the model's personality (Gen Z, slang, emoji) and the required respectful/sensitive tone towards holy texts. This defines *how the model must act*. |
| **Chat History** | **Memory** | Includes previous dialogue (user/AI). The model uses this to maintain context and give consistent answers. |
| **Context** | **Information Source** | The 25 pieces of Meal/Tafsir retrieved using the MMR algorithm from the Vector Database. The model must base its answer **only** on these texts. |
| **User Query (Input)** | **Current Request** | The user's specific question at that moment. |

---

## 💡 Answer Generation Logic (RAG Workflow)

### 🖼️ RAG Diagram: Retrieval + Generation

The "brain" of our project (RAG) works by following the **6 main steps**. Here's which component handles which step:

| **Diagram Step** | **Process** | **Technology / Parameter Used** | **Explanation** |
|------------------|-------------|----------------------------------|-----------------|
| **1. Query Embedding** | The user's text question is taken. | **BGE-M3 (Embedding Model)** | The query is converted into a numerical vector that the RAG system can understand. |
| **2. Vector Representation** | The conversion is complete. | **Vector** | The question is now a ready coordinate point for searching the **Vector Database (Chroma DB)**. |
| **3. Database Search** | The query vector is sent to the database. | **Chroma DB** | The database pulls the first **60 candidate chunks (MMR_FETCH_K)** that are most likely relevant to the query. |
| **4. Relevance Retrieval (Top-k Chunks)** | The MMR Algorithm runs. | **MMR (MMR_K=25, LAMBDA=0.5)** | Out of the 60 candidate chunks, the **25 chunks** with the highest relevance and subject diversity are selected. |
| **5. Generation Input** | The Top-k chunks are sent to the LLM. | **Gemini 2.5 Flash** | The **25 chunks** and the **SYSTEM_INSTRUCTION** are combined, instructing the LLM to formulate an answer. |
| **6. Answer** | The LLM generates the response. | **LLM Output** | The model summarizes the chunks and adds its own commentary in the Gen Z tone (AI Commentary) to structure the final answer. |

The LLM uses the **25 chunks** (Meal, Tafsir-Ayat, Tafsir-Surah) to structure the answer in three main parts:

| **Answer Section** | **LLM's Source Usage** | **Purpose (Vibe)** |
|---------------------|------------------------|--------------------|
| **1. Quick Summary Answer** | Generated using information from all 25 chunks (Meal + Tafsir). | To give a **quick and clear** introduction to the topic asked. |
| **2. AI Commentary & Analysis** | Uses **Tafsir-Ayat** and **Tafsir-Surah** chunks. | Deepens the topic, adds context and interpretation. The model provides its Gen Z commentary (slang, emoji) creatively in this part. |
| **3. References** | Uses **only Meal** type chunks. | To prove that the answer is based on the **direct text** of the Quran (Meal) and show respect. |

---

## 🧠 How It Works (Architecture)

The application structure (architecture) is quite **lit**:

| **Component** | **Role** | **Technology** |
|----------------|-----------|----------------|
| **Large Language Model (LLM)** | Answer generation, tone adjustment, interpreting the RAG output. | **See Quick Technical Overview** 💡 |
| **Vector Database (Vector DB)** | Storing the Quran meaning and commentary texts and fast searching. | **Chroma DB** (Data is uploaded as a ZIP.) |
| **Embedding Model** | Converting questions and texts into numerical vectors. | **See Quick Technical Overview** 🇹🇷 |
| **Application Framework** | Managing the user interface (UI) and the RAG chain. | **Gradio (Python)** 🖥️ |

### ⚙️ Maximum Marginal Relevance (MMR) Parameters

| **Parameter** | **Value** | **Description** |
|----------------|-----------|-----------------|
| **MMR_FETCH_K** | 60 | The number of candidate documents initially fetched from the database. |
| **MMR_K** | 25 | The final number of documents selected from the 60 candidates using the diversity algorithm. |
| **MMR_LAMBDA_MULT** | 0.5 | The balance factor between Diversity and Relevance. |

---

## 📂 Project Structure (File Structure)

The core files and their roles in this repository are listed below.  
**The `chroma_db_final.zip` and `processed_kuran_documents.json` files are automatically downloaded from Hugging Face when the application runs.**
```
.
├── app.py
├── requirements.txt
├── chroma_db_final.zip
├── processed_kuran_documents.json
└── README.md

```

| **File Name** | **Role and Vibe** |
|----------------|-------------------|
| `app.py` | 🧠 **Brain:** Contains the entire chatbot logic (LLM, RAG chain, Gradio interface) and the **SYSTEM_INSTRUCTION** defining the Gen Z tone. |
| `requirements.txt` | 🛠️ **Dependencies:** Lists the Python libraries required for the project to run. |
| `chroma_db_final.zip` | 💾 **Knowledge Base (Compressed):** The ZIP archive of the **Chroma Vector Database**. |
| `processed_kuran_documents.json` | 📄 **Raw Data:** The raw JSON list of the Meal and Tafsir texts, with added metadata. |
| `README.md` | 📜 **Vibe Check:** The summary and installation instructions you are currently reading. |

---

## 🔍 Data Source Details (`processed_kuran_documents.json`)

This file is the primary data source, created by scraping and chunking various sources for RAG.

Each data chunk (Document) is stored with its content (`page_content`) and metadata.  
The data includes **3 different source types**:

| **Source Type** | **Description** |
|------------------|-----------------|
| `Meal` | Simplified Turkish meaning of the Ayats (verses). **Shown directly as a reference in the RAG output.** |
| `Tafsir-Ayat` | Commentary texts explaining the context, reasons for revelation, and deep meanings of a single Ayat. |
| `Tafsir-Surah` | Commentary texts summarizing the **general theme and main message of the entire Surah**. |


#### Example Data Chunks:

```
// Tip 1: Meal (Kısa ve net ayet metni)
{
  "page_content": "Doğruysanız kitabınızı getirin!",
  "metadata": {
    "sure_no": 37,
    "sure_name": "Saffat",
    "ayet_no": 157,
    "kaynak_tipi": "Meal"
  }
}

// Tip 2: Tefsir-Ayet (Tek ayetin detaylı yorumu)
{
  "page_content": "Konuyla ilgili benzer bir ifade En'âm 6:145'te şu şekildedir: ... Yüce Allah yiyecek, içecek vs. şeylerle ilgili olarak insanların bazen kendi iradeleri dışında gelişebilecek birtakım durumları da hatırlatmaktadır. ...",
  "metadata": {
    "sure_no": 16,
    "sure_name": "Nahl",
    "ayet_no": 115,
    "kaynak_tipi": "Tafsir-Ayat"
  }
}

// Tip 3: Tefsir-Sure (Tüm surenin ana teması ve özeti)
{
  "page_content": "5. âyetten itibaren, "...kim, (malını Allah rızası için) verir, muttaki (duyarlı) davranır ve en güzeli (tevhidi, vahyi, cenneti) tasdik ederse, kolay olanı ona (daha da) kolaylaştıracağız." ifadeleriyle ilahi iradenin insan tercihleriyle ilgili tutumu ortaya konulmaktadır. ... Sûrenin sonunda, kişinin dünya hayatında yaptığı fedakârlıkların ve arınmak amacıyla gerçekleştirdiği çalışmaların, sadece Yüce Allah'ın rızasına yönelik olması gerektiği özellikle vurgulanmaktadır.",
  "metadata": {
    "sure_no": 92,
    "sure_name": "Leyl",
    "ayet_no": "Sure Bazlı",
    "kaynak_tipi": "Tafsir-Sure"
  }
}

```

### 🛠️ Local Installation (If Required)

You can follow these steps to run this project on your local computer.

#### 1\. Prerequisites

-   **Python 3.9 or higher** must be installed.

-   Download the project files and navigate to the main directory (e.g., using `git clone`).

-   **IMPORTANT:** The main data files, `chroma_db_final.zip` (Vector DB) and `processed_kuran_documents.json` (Raw Data), must be downloaded from the project's Hugging Face repository and placed in your main directory.

-   Using a **virtual environment** (`venv`, `conda`, etc.) is highly recommended.

#### 2\. Install Dependencies

Install all necessary Python libraries (from the `requirements.txt` file):

```
pip install -r requirements.txt

```

#### 3\. Set the API Key (Very Important!)

The application requires a Gemini API Key for the LLM and Embeddings models. You must set your key as an environment variable named **`GEMINI_API_KEY`**.

```
# Linux/macOS
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# Windows (Command Prompt)
set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

```

#### 4\. Start the Application

After all installations are complete, run the main application. The application will automatically extract the Chroma DB ZIP file (which you must have downloaded in Step 1) and load the database:

```
python app.py

```

When the application starts successfully, the Gradio interface will open in your browser. Please check that the "System Status" box that appears in the browser is **READY**.
