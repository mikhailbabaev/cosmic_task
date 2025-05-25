# About project
This project deploys a numerical model and connects a tool to get the current time.
The model used is **llama3.2** from **Ollama**.

The **LangChain** framework is used to work with the local model.
Specifically, an agent of type `ZERO_SHOT_REACT_DESCRIPTION` is applied, which can call the `get_current_time` tool when asked about the time.

The model is not perfect and gives the correct answer to the question **What time is it?** roughly one time out of three to four attempts.

---

# Installation and launch
To run correctly, please follow these steps:

1. **Install Ollama**

If you don't have Ollama installed,
go to the official Ollama website and download the application for your OS.

2. **Pull the llama3.2 model**

Run the following command in your terminal:
```bash
ollama pull llama3.2
```

3. **Clone the repository and install dependencies**
```bash
git clone https://github.com/mikhailbabaev/cosmic_task.git
cd cosmic_task
python -m venv .venv
source .venv/bin/activate   # for Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. **Run project**
```bash
langgraph dev
```

# p.s

You can run script in terminal to Local testing without langsmith
```bash
python main.py
```
