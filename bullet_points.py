import streamlit as st

def bullet_points_ui(llm, retriever):
    with st.expander("📋 Break Document into Bullet Points"):
        if st.button("Generate Bullet Points"):
            all_text = " ".join([doc.page_content for doc in retriever.vectorstore.docstore._dict.values()])
            prompt = (
                "Break down the following document into detailed, well-organized bullet points. "
                "Cover all key ideas, facts, and sections so that someone can understand the entire document just by reading these points.\n\n"
                "Make the bullet points as detailed as possible, they should be comprehensive and cover all aspects"
                "Consider an example: of a chapter from a maths textbook for a real numbers chapter, you should not only mention the theorem that states root 2 is irrational , but also its explanation and the entire proof."
                f"Document:\n{all_text}\n\nBullet Points:"
            )
            result = llm.invoke(prompt)
            # Extract only the content if needed
            if isinstance(result, dict) and "content" in result:
                bullet_text = result["content"]
            elif hasattr(result, "content"):
                bullet_text = result.content
            elif hasattr(result, "text"):
                bullet_text = result.text
            else:
                bullet_text = str(result)
            # Display as markdown for nice formatting
            st.markdown(bullet_text)