import streamlit as st
from vector_stores import get_data, load_documents, retrieve_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Dict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()

class OutputSchema(BaseModel):
    claim: Annotated[str, Field(description="The original claim input by the user. Please ensure this is exactly as the user provided it.")]
    #rating: Annotated[Literal["True", "False", "Miscaptioned", "Mostly True", "Research in Progress", "Labeled Satire", "Unproven"], Field(description="Feedback on the claim input by the user. Must be one of True, False, Miscaptioned, Mostly True, Research in Progress, Labeled Satire, Unproven")]
    probabilities: Dict[
        Literal[
            "True", "False", "Unproven"
        ],
        float
    ] = Field(description="A probability score (0-1) for each category. Scores should sum to 1.")
    Rationale: Annotated[str, Field(description="A detailed explanation behind the assignment of these probabilities. Should reference specific pieces of evidence from the context.")]

output_parser = PydanticOutputParser(pydantic_object=OutputSchema)

format_instruction = output_parser.get_format_instructions()

#model
model = ChatOpenAI(model="gpt-4o")

#load data
data = get_data("technology_fact_checks.json")
documents = load_documents(data)

#load retriever
retriever = retrieve_context(documents)

st.title("📰 Claim Verification Assistant")


##Added part
st.set_page_config(
    page_title="Truth Seeker - Fact Checker",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .result-true { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; }
    .result-false { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; }
    .result-unproven { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)


##Added part

user_claim = st.text_area("Enter a claim to fact-check:", height=100)

if st.button("Check Claim"):
    if user_claim.strip():
        with st.spinner("Retrieving context and verifying..."):
            # retrieve relevant docs only
            docs = retriever.invoke(user_claim)
            context = "\n".join([doc.page_content for doc in docs])

            template = """
                        You are a claim fact-checking assistant. Your task is to evaluate the truthfulness of a given claim based on relevant information retrieved from trusted news sources as provided in the context below.

                        Context: {context}

                        Claim: {query}

                        {format_instruction}
                        """
            prompt = PromptTemplate(template=template, input_variables=["context", "query"], partial_variables={"format_instruction": format_instruction})

            chain = prompt | model | output_parser

            result = chain.invoke({"context": context, "query": user_claim})

            # pick category with highest probability
            best_category = max(result.probabilities, key=result.probabilities.get)
            confidence = result.probabilities[best_category]
        
        st.subheader("✅ Fact-Check Result")
        st.markdown(f"<h3 style='font-size: 1.5em;'>Claim: {result.claim}</h3>", unsafe_allow_html=True)
        #st.markdown(f"**Claim:** {result.claim}")

        # Added part
        #result_class = "result-true" if best_category == "True" else "result-false" if best_category == "False" else "result-unproven"
        #st.markdown(f'<div class="{result_class}"><strong>Verdict: {best_category}</strong> ({confidence:.2%} confidence)</div>', unsafe_allow_html=True)
        result_class = "result-true" if best_category == "True" else "result-false" if best_category == "False" else "result-unproven"
        st.markdown(f'<div class="{result_class}"><h2 style="font-size: 1.8em; margin: 0;">Verdict: {best_category}</h2><p style="font-size: 1.2em; margin: 5px 0 0 0;">Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
        #st.markdown(f"**Predicted Category:** {best_category} ({confidence:.2%} confidence)")
        #Added part
        # ---------- Plot probability distribution ----------
        st.subheader("📊 Probability Distribution")

        df = pd.DataFrame({
            "Category": list(result.probabilities.keys()),
            "Probability": list(result.probabilities.values())
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["Category"], df["Probability"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Model-assigned probabilities per category")
        plt.xticks(rotation=30, ha="right")

        st.pyplot(fig)

        #st.markdown(f"**Rationale:** {result.Rationale}")
        st.markdown(f"<p style='font-size: 1.2em; line-height: 1.6;'><strong>Rationale:</strong> {result.Rationale}</p>", unsafe_allow_html=True)

        st.subheader("🔗 Top Sources")
        for i, doc in enumerate(docs, start=1):
            url = doc.metadata.get("url", None)
            title = doc.metadata.get("title", f"Source {i}")
            if url:
                st.markdown(f"{i}. [{title}]({url})")
            else:
                st.markdown(f"{i}. {title}")

    else:
        st.warning("Please enter a claim to fact-check.")