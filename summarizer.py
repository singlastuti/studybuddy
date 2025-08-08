import streamlit as st

def _extract_text(result):
    return getattr(result, "content", getattr(result, "text", str(result)))

def summarizer_ui(llm, retriever):
    with st.expander("Summarise the uploaded document"):
        summary_length = st.radio(
            "Select summary length:",
            ("Concise", "Medium", "Detailed"),
            horizontal=True,
        )
        if st.button("Summarise Document"):
            with st.spinner("Summarising document…"):
                length_map = {
                "Concise": "in not more than 2-3 sentences",
                "Medium": "in a short paragraph",
                "Detailed": "in a detailed summary, explaining each key point in detail",
                }
                summary_instruction = (
                    f"Summarise the uploaded document {length_map[summary_length]}. "
                    "Use only the document content. If information isn't present, say so."
                )

                # Prefer chunks saved in session (avoids reaching into private attrs)
                chunks = st.session_state.get("chunks", [])
                if not chunks:
                    # As a fallback, retrieve a broad set and deduplicate
                    docs = retriever.vectorstore.similarity_search("overview", k=20)
                    chunks = docs

                # Limit the number of chunks to control token usage
                chunks = chunks[:20]

                # Map step: summarise each chunk
                partial_summaries = []
                for doc in chunks:
                    part_prompt = (
                        "Summarise this excerpt from the document. Be faithful to the text.\n\n"
                        f"Excerpt:\n{doc.page_content}\n\nBrief Summary:"
                    )
                    res = llm.invoke(part_prompt)
                    partial_summaries.append(_extract_text(res))

                # Reduce step: combine partial summaries
                combined = "\n".join(partial_summaries)
                reduce_prompt = (
                    f"{summary_instruction}\n\nHere are partial summaries from different parts of the document. Combine them into a cohesive final summary without repeating points.\n\n"
                    f"Partial Summaries:\n{combined}\n\nFinal Summary:"
                )
                final_res = llm.invoke(reduce_prompt)
                st.info(_extract_text(final_res))