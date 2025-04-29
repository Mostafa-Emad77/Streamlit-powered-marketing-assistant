# 34ML AI Assistant

A **Streamlit‑powered marketing assistant** that crawls a website, builds a private knowledge base, and helps you craft tailored posts & images for LinkedIn, Facebook, Instagram, blogs, and email—all from one simple chat‑like UI.

---

## ✨ Features

- **Automatic site crawl** – Uses Firecrawl to scrape up to 50 pages (default: `https://34ml.com/`) and store them locally.
- **Structured knowledge‑base generation** – Sends the crawl to Google Gemini to create a rich Markdown knowledge base.
- **Multi‑channel content creation** – One‑click generation of posts for LinkedIn, Facebook, Instagram, long‑form blogs, or email newsletters, each with platform‑specific tone and length.
- **Image generation** – Leverages Hugging Face `HiDream-I1` to create on‑brand visuals.
- **Conversation memory** – Remembers your previous prompts so follow‑ups stay in context.
- **Content history & approval workflow** – Saves every generated post to `content_history.json`; approve or decline inside the UI.

---

## 🛠 Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10 or newer | Recommended to use a virtual environment |
| **API keys** | Sign up and put these in a `.env` file:<br/>`FIRECRAWL_KEY` – Firecrawl<br/>`GEMINI_API_KEY` – Google AI Studio<br/>`HF_API_KEY` – Hugging Face Inference |

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your‑org/34ml‑ai‑assistant.git
cd 34ml‑ai‑assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env  # then edit .env with your keys
```

---

## ▶️ Running the App

```bash
streamlit run Scrape_34_ml.py
```

Open the provided local URL (usually `http://localhost:8501`) in your browser.

---

## 🧑‍💻 Usage Tips

1. **Select a platform** from the dropdown (LinkedIn, Facebook, etc.).
2. **Pick a content type** (e.g. “Visual Success Story” for Instagram).
3. **Chat** – e.g. “Write an intro post announcing our new AI integration.”
4. **Approve / Decline** generated content; approved posts are logged.
5. Use the **Image Generation** box for tailored visuals (prompt‑based).

---

## ⚙️ Environment‑variable example (`.env`)

```dotenv
FIRECRAWL_KEY=fc_********************************
GEMINI_API_KEY=gai_********************************
HF_API_KEY=hf_********************************
TARGET_URL=https://34ml.com/  # optional override
MAX_PAGES=50                  # optional override
```

---

## 🏗 How It Works (Architecture)

1. **Firecrawl** → fetches site → `site_content.txt`
2. **Gemini** → converts raw scrape to structured Markdown (`34ml_analysis.md`)
3. **Streamlit UI** → loads knowledge base + conversation memory
4. **Gemini (chat)** → platform‑aware content generation
5. **Hugging Face** → optional image generation

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue to discuss what you want to change.

---


