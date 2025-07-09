import streamlit as st

def summarizer_ui(llm, retriever):
    with st.expander("Summarise the uploaded document"):
        summary_length = st.radio(
            "Select summary length:",
            ("Concise", "Medium", "Detailed"),
            horizontal=True
        )
        if st.button("Summarise Document"):
            # Map summary length to instructions
            length_map = {
                "Concise": "in not more than 2-3 sentences",
                "Medium": "in a short paragraph",
                "Detailed": "in a detailed summary, explaining each key point in detail"
            }
            summary_instruction = (
                f"Summarise the uploaded document {length_map[summary_length]}. "
                "Use only the document content."
            )
            # Use the retriever to get all chunks (or just concatenate all text)
            all_text = " ".join([doc.page_content for doc in retriever.vectorstore.docstore._dict.values()])
            # Ask the LLM to summarise
            summary_prompt = (
                f"{summary_instruction}\n\nDocument:\n{all_text}\n\nSummary:"
            )
            summary_result = llm.invoke(summary_prompt)
            # If summary_result is a dict or has a 'content' or 'text' attribute, extract it:
            if isinstance(summary_result, dict) and "content" in summary_result:
                summary_text = summary_result["content"]
            elif hasattr(summary_result, "content"):
                summary_text = summary_result.content
            elif hasattr(summary_result, "text"):
                summary_text = summary_result.text
            else:
                summary_text = str(summary_result)

            st.info(summary_text)