from groq import Groq
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_llm_score(text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You will receive text that you are required to determine whether it is AI generated or human. Assess the text and return a score based on how confident you are. The score must be between 0 and 1 where 1 means definitely AI and 0 means definitely human. Do not include any text. Only the numerical score."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    raw = response.choices[0].message.content.strip()
    return float(raw)

def get_stylometric_score(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    lengths = [len(s.split()) for s in sentences]
    if len(lengths) > 1:
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    else:
        variance = 0

    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) > 0:
        ttr = len(set(words)) / len(words)
    else:
        ttr = 0

    normalized_variance = variance / (variance + 10)
    variance_score = 1 - normalized_variance
    ttr_score = 1 - ttr

    return round((variance_score + ttr_score) / 2, 3)