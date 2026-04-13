from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 🧠 قاعدة المعرفة
knowledge = [
    "بوكيه حب رومانسي يحتوي على ورد الجوري الأحمر.",
    "الورد الأحمر يرمز للحب والرومانسية.",
    "بوكيه التخرج يحتوي على ألوان مبهجة مثل الأصفر.",
    "الورد الأبيض مناسب للأعراس والمناسبات الهادئة."
]

# 🔍 تجهيز البحث
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(knowledge)

class Request(BaseModel):
    question: str

@app.post("/ask")
def ask(req: Request):
    question_vec = vectorizer.transform([req.question])

    similarity = cosine_similarity(question_vec, X)
    best_match_index = similarity.argmax()

    answer = knowledge[best_match_index]

    return {"answer": answer}
