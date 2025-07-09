import streamlit as st

def quiz_generator_ui(llm, retriever):
    with st.expander("📝 Generate a Quiz from the uploaded document"):
        quiz_type = st.selectbox(
            "Select quiz type:",
            ("MCQs", "Fill-in-the-blank", "Flashcards")
        )
        num_questions = st.slider("Number of questions/cards:", 1, 10, 5)

        if st.button("Generate Quiz"):
            quiz_instruction_map = {
                "MCQs": f"Generate {num_questions} multiple-choice questions (with 4 options and the correct answer marked) from the uploaded document.",
                "Fill-in-the-blank": f"Generate {num_questions} fill-in-the-blank questions (with answers) from the uploaded document.",
                "Flashcards": f"Generate {num_questions} flashcards (with question and answer pairs) from the uploaded document."
            }
            quiz_instruction = quiz_instruction_map[quiz_type]
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

            st.info(quiz_text)