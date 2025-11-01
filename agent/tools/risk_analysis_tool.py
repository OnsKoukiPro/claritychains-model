from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI as ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


def create_risk_analysis_chain(retriever):
    """
    Creates a chain for generating a risk analysis.
    """
    llm = ChatOpenAI(
        azure_deployment="ClarityChain",  # or your deployment
        api_version="2024-08-01-preview",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    prompt_template = """You are an assistant for analyzing procurement documents.
Use the following pieces of context to answer the question at the end.
The context contains text from procurement documents. Each document corresponds to one supplier.
Your task is to create a risk analysis for the suppliers based on the provided document content.

The output should be a markdown table for the summary, and then a detailed breakdown for each vendor.
Follow this format exactly:

**Risk Analysis Summary**

| Vendor        | Quoted Price   | Total TCO      | Lead Time | Risk Score | Risk Level |
|---------------|----------------|----------------|-----------|------------|------------|
| Supplier A    | € value        | € value        | X days    | score/100  | Low/Medium/High |
| Supplier B    | € value        | € value        | Y days    | score/100  | Low/Medium/High |

**Risk Score Breakdown**

The risk score is out of 100, where a **higher score means higher risk**.
- **Positive points (+)** must be assigned to factors that **increase risk**. Examples: higher price, longer lead time, unclear terms, lack of documentation, negative references.
- **Negative points (-)** must be assigned to factors that **decrease risk**. Examples: lower price, faster delivery, proven track record, good references, compliance with standards.

Here is an example of how to structure the breakdown:

**Example Supplier (Risk Score: 20/100)**
* +15 points: Moderate schedule risk (46 weeks lead time)
* +10 points: Price variance €270k above base cost
* -5 points: Proven supplier track record in region
* = 20 Total Risk Score

Now, generate the risk analysis for the suppliers from the provided documents.

You MUST determine the supplier names from the document content.
You MUST calculate a risk score for each supplier (out of 100, higher is riskier).
You MUST provide a breakdown of the factors contributing to the risk score, following the scoring logic explained above.
For Total TCO, if operational/maintenance costs are not specified, assume TCO is the same as the quoted price and state this assumption.

{context}

Question: {question}

Return ONLY the data in the format described above, and nothing else.
If a category is not applicable or data is not found in the provided documents, you must write 'Not found' or 'N/A'. Do not leave any cells empty.
"""
    RISK_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    risk_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RISK_PROMPT},
    )

    return risk_chain
