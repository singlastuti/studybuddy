import streamlit as st
import re

def _extract_text(result):
    return getattr(result, "content", getattr(result, "text", str(result)))

def quiz_generator_ui(llm, retriever):
    with st.expander("📝 Generate a Quiz from the uploaded document"):
        num_questions = st.slider("Number of questions:", 1, 10, 5)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz…"):
                chunks = st.session_state.get("chunks", [])
                if not chunks:
                    docs = retriever.vectorstore.similarity_search("key concepts", k=15)
                    chunks = docs

            # Build a compact content basis by sampling up to ~10 chunks
                sample = chunks[:10]
                basis = "\n\n".join(d.page_content for d in sample)

                quiz_instruction = (
                f"Using only the document content below, generate {num_questions} fill-in-the-blank questions. "
                "Provide the answers immediately after each question in the format 'Answer: ...'. "
                "Ensure questions span different parts of the content and avoid trivial blanks."
            )
                quiz_prompt = (
                f"{quiz_instruction}\n\nDocument Content:\n{basis}\n\nQuiz:"
            )
                quiz_text = _extract_text(llm.invoke(quiz_prompt))

            # Show questions with collapsible answers
                qa_pairs = re.split(r"\n(?=\d+\.)", quiz_text.strip())
                for pair in qa_pairs:
                    lines = [ln for ln in pair.strip().split("\n") if ln]
                    if not lines:
                        continue
                    question = lines[0]
                    # Find an explicit 'Answer:' line or fallback to second line
                    answer_line = next((ln for ln in lines[1:] if ln.lower().startswith("answer:")), lines[1] if len(lines) > 1 else "Answer: (not detected)")
                    st.markdown(f"**{question}**")
                    with st.expander("Show Answer"):
                        st.markdown(answer_line)
                    st.markdown("---")