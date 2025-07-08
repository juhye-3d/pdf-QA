# streamlit_pdf_rag_tfidf.py

import streamlit as st
import requests
import io
from PyPDF2 import PdfReader
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 키 입력
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PDF 텍스트 추출
def get_pdf_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with io.BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        return f"[PDF 불러오기 실패] {e}"

# 슬라이딩 분할
def split_text(text, chunk_size=3000, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# TF-IDF 기반 유사 chunk 추출
def get_most_similar_chunks(query, chunks, top_n=2):
    corpus = chunks + [query]  # 전체 문서 chunk + 질문
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])  # 질문 vs chunks
    sim_scores = cosine_sim.flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return [chunks[i] for i in top_indices]

# GPT 응답 스트리밍
def stream_gpt_response(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 신뢰도 높은 문서 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.3,
        stream=True
    )
    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.get("content", "")
        full_response += delta
        yield delta
    return full_response

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF GPT Q&A (with TF-IDF)", layout="centered")
st.title("📄 PDF 기반 GPT Q&A (TF-IDF 기반 Chunk 추출)")

pdf_url = st.text_input("🔗 PDF 링크를 입력하세요", 
    value="https://www.themoonlight.io/file?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2501.00539")

# PDF 로딩
if st.button("📥 PDF 불러오기"):
    with st.spinner("PDF에서 텍스트를 추출 중입니다..."):
        full_text = get_pdf_text_from_url(pdf_url)

    if full_text.startswith("[PDF 불러오기 실패]"):
        st.error(full_text)
    else:
        st.success("✅ PDF 문서 불러오기 완료!")
        st.text_area("📄 일부 미리보기", full_text[:1000], height=200)

        # chunk 분할
        chunks = split_text(full_text)
        st.write(f"🔍 총 {len(chunks)}개 chunk로 분할됨")

        # 질문
        question = st.text_input("❓ GPT에게 질문하세요", value="이 논문의 핵심은 무엇인가요?")

        if st.button("🔍 질문하기"):
            with st.spinner("질문과 관련 있는 chunk를 찾는 중..."):
                top_chunks = get_most_similar_chunks(question, chunks, top_n=2)
                context = "\n\n".join(top_chunks)
                prompt = f"다음 문서를 바탕으로 질문에 답해주세요:\n\n{context}\n\n질문: {question}"

            st.markdown("#### 💬 GPT 응답:")
            placeholder = st.empty()
            result = ""
            for chunk in stream_gpt_response(prompt):
                result += chunk
                placeholder.markdown(result)
