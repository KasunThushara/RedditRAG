
# RedditRAG with Gemini API 🔍🤖

This is a **Retrieval-Augmented Generation (RAG)** tool that fetches Reddit posts using the Reddit API and allows you to ask questions about them using the **Google Gemini API**.

---

## 📁 Features

- Fetch Reddit post contents from `.txt` link lists
- Create vector database for semantic search
- Ask contextual questions from the content using Gemini
- Built with **Streamlit** for a simple interactive UI

---

## 🚀 Quick Start (Windows + PyCharm)

### 🔽 1. Download & Extract

1. Go to [GitHub Repo](https://github.com/KasunThushara/RedditRAG)
2. Click the green `<> Code` button
3. Choose `Download ZIP`
4. Extract it to any folder (e.g., `D:\Projects\RedditRAG`)

### 🧠 2. Open in PyCharm

1. Open **PyCharm**
2. Choose **Open** and select the extracted folder

### 🐍 3. Set Up Virtual Environment (Optional but Recommended)

1. Go to `File > Settings > Project: RedditRAG > Python Interpreter`
2. Click ⚙ > `Add...` > Choose `Virtualenv Environment`
3. Select your Python interpreter (e.g., Python 3.11) and click OK

### 📦 4. Install Dependencies

Use PyCharm Terminal:

```bash
pip install -r requirements.txt
pip install protobuf==3.20.*
````

### ▶ 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`)

---

## 🔑 Configuration

You'll need:

* **Reddit API credentials** (Client ID, Client Secret, User Agent)
* **Google Gemini API Key**

You can enter these values directly in the UI.

---

## 📂 Folder Input

Provide the path to a folder that contains `.txt` files with Reddit post links — each file can contain one or multiple Reddit URLs.

