import os
import streamlit as st
import requests
import io
import pdfplumber
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ PDF 추출 함수 ------------------
def get_pdf_text_with_plumber(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("application/pdf"):
            return "[PDF 불러오기 실패] 이 링크는 PDF 파일이 아닙니다."

        with io.BytesIO(response.content) as f:
            with pdfplumber.open(f) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
    except Exception as e:
        return f"[PDF 불러오기 실패] {e}"

# ------------------ Chunk 분할 함수 ------------------
def split_text(text, chunk_size=3000, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# ------------------ TF-IDF 유사 chunk 추출 ------------------
def get_most_similar_chunks(query, chunks, top_n=2):
    corpus = chunks + [query]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    sim_scores = cosine_sim.flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return [chunks[i] for i in top_indices]

# ------------------ GPT 응답 스트리밍 ------------------
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

# ------------------ Streamlit 시작 ------------------

st.set_page_config(page_title="📄 PDF GPT Q&A", layout="centered")
st.title("📄 PDF 기반 GPT Q&A (최대 3회 질문)")

# 🔁 상태 초기화
if "qa_round" not in st.session_state:
    st.session_state.qa_round = 1
    st.session_state.chat_history = []
    st.session_state.chunks = []
    st.session_state.document_loaded = False

# 📥 PDF URL 입력
pdf_url = st.text_input("🔗 PDF 링크를 입력하세요", 
    value="https://arxiv.org/pdf/2501.00539.pdf")

if st.button("📥 PDF 불러오기"):
    with st.spinner("📄 PDF에서 텍스트 추출 중..."):
        full_text = get_pdf_text_with_plumber(pdf_url)

    if full_text.startswith("[PDF 불러오기 실패]"):
        st.error(full_text)
    else:
        st.success("✅ PDF 문서 불러오기 완료")
        st.session_state.chunks = split_text(full_text)
        st.session_state.qa_round = 1
        st.session_state.chat_history = []
        st.session_state.document_loaded = True
        st.text_area("📄 문서 미리보기", full_text[:1000], height=200)
        st.info(f"📌 총 {len(st.session_state.chunks)}개 chunk로 분할됨")

# 🔎 질문 입력 & GPT 응답 (최대 3회)
if st.session_state.document_loaded and st.session_state.qa_round <= 3:

    question = st.text_input(
        f"❓ 질문 {st.session_state.qa_round}: GPT에게 질문하세요", 
        key=f"question_{st.session_state.qa_round}"
    )

    if st.button(f"🔍 질문 {st.session_state.qa_round} 실행"):
        with st.spinner("질문과 관련 있는 chunk를 찾는 중..."):
            top_chunks = get_most_similar_chunks(question, st.session_state.chunks, top_n=2)
            context = "\n\n".join(top_chunks)
            prompt = f"다음 문서를 참고하여 질문에 답해주세요:\n\n{context}\n\n질문: {question}"

        st.markdown(f"#### 💬 GPT 응답 {st.session_state.qa_round}:")
        placeholder = st.empty()
        response = ""
        for chunk in stream_gpt_response(prompt):
            response += chunk
            placeholder.markdown(response)

        # 기록 저장
        st.session_state.chat_history.append((question, response))
        st.session_state.qa_round += 1

# ⛔ 질문 제한 초과 안내
if st.session_state.qa_round > 3:
    st.info("✅ 최대 3번까지 질문이 가능합니다. 페이지를 새로고침하면 다시 시작할 수 있어요.")

# 📜 이전 대화 보기
if st.session_state.chat_history:
    with st.expander("📚 이전 질문과 응답 보기"):
        for i, (q, a) in enumerate(st.session_state.chat_history, start=1):
            st.markdown(f"**Q{i}: {q}**")
            st.markdown(f"**A{i}: {a}**")
