import streamlit as st
import re

def quiz_generator_ui(llm, retriever):
    with st.expander("📝 Generate a Quiz from the uploaded document"):
        num_questions = st.slider("Number of questions:", 1, 10, 5)

        if st.button("Generate Quiz"):
            quiz_instruction = f"Generate {num_questions} fill-in-the-blank questions (with answers) from the uploaded document."
            all_text = " ".join([doc.page_content for doc in retriever.vectorstore.docstore._dict.values()])
            quiz_prompt = (
                f"{quiz_instruction}\n\nDocument:\n{all_text}\n\nQuiz:"
            )
            quiz_result = llm.invoke(quiz_prompt)
            # Extract only the content if needed
            if isinstance(quiz_result, dict) and "content" in quiz_result:
                quiz_text = quiz_result["content"]
            elif hasattr(quiz_result, "content"):
                quiz_text = quiz_result.content
            elif hasattr(quiz_result, "text"):
                quiz_text = quiz_result.text
            else:
                quiz_text = str(quiz_result)

            # Show fill-in-the-blank questions with answers in expanders
            qa_pairs = re.split(r"\n(?=\d+\.)", quiz_text)
            for pair in qa_pairs:
                lines = pair.strip().split("\n")
                if len(lines) >= 2:
                    question = lines[0]
                    answer = lines[1]
                    st.markdown(f"**{question}**")
                    with st.expander("Show Answer"):
                        st.markdown(answer)
                    st.markdown("---")