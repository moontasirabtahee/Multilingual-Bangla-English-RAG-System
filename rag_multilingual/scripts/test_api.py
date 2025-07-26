import requests

def test():
    url = "http://localhost:7860/rag/query"
    test_cases = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    for q in test_cases:
        resp = requests.post(url, json={"query": q})
        print("Q:", q)
        print("A:", resp.json()["answer"])
        print("Eval:", resp.json()["evaluation"])
        print("---")

if __name__ == "__main__":
    test()