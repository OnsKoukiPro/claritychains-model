import traceback
import json
import os
import re
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Import tools
from tools.file_tools import file_reading_tools
from tools.currency_converter_tool import currency_converter
from tools.calculator_tool import calculator_tool
from tools.retriever import Retriever


def _clean_json_string(json_str):
    """
    Removes markdown code blocks and other non-JSON characters from a string.
    """
    # Find the JSON object within the string
    match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if match:
        return match.group(0)
    return json_str


def _get_llm():
    """Get configured Azure OpenAI LLM instance"""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
        model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o"),
    )


def create_gap_analysis_agent():
    """
    Creates an agent that performs a detailed gap analysis based on provided text.
    """
    llm = _get_llm()

    detailed_category_instructions = """
    - **Total Cost of Ownership (TCO):** Extract the total quoted price. If the documents provide details on operational/maintenance/logistics costs, add them to the quoted price to calculate the TCO. If not, you MUST state that TCO is assumed to be the same as the quoted price in the 'Gap Analysis' column.
    - **Payment Terms:** Detail the payment terms (e.g., net-30, net-90). Note any flexibility or discounts for early payment.
    - **Price Stability:** Note any clauses related to price stability, such as fixed-price terms or validity periods.
    - **Lead Time:** Extract the delivery lead time.
    - **Technical Specifications:** Summarize the key technical specifications of the offered product/service.
    - **Certifications:** List any mentioned industry or regulatory certifications (e.g., ISO, CE).
    - **Incoterms:** Identify the Incoterms (e.g., FOB, DDP) which clarify shipping and risk responsibilities.
    - **Warranty:** State the provided warranty terms and duration.
    - **Risks & Deviations:** Summarize any commercial or technical risks, or any deviations from standard expectations or requirements.
    """

    template = """You are a specialist in procurement gap analysis.
Given the text of a single procurement offer, you must extract the information for each category below and create a detailed analysis.

**Offer Text:**
{offer_text}

**Analysis Categories:**
{detailed_category_instructions}

**Output Format:**
Return a JSON object with a "detailed_gap_analysis" key.
The value should be another JSON object with "headers" and "rows".
The headers should be: ["Category", "Details", "Gap Analysis"]
For each category, create a row with the extracted details and a brief analysis of any gaps or key points.

**Example Output:**
{{
  "detailed_gap_analysis": {{
    "headers": ["Category", "Details", "Gap Analysis"],
    "rows": [
      ["Total Cost of Ownership (TCO)", "$1,200,000", "Price is higher than typical market rates."],
      ["Payment Terms", "Net 60", "Standard terms, no major gaps."]
    ]
  }}
}}

Begin!

**Analysis:**
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["offer_text"],
        partial_variables={
            "detailed_category_instructions": detailed_category_instructions
        },
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def create_risk_analysis_agent(risk_weights):
    """
    Creates an agent that performs a risk analysis on a procurement offer.
    """
    llm = _get_llm()

    risk_dimensions_instructions = """
**Core Risk Dimensions (Score 1-5, 1=Low Risk, 5=High Risk):**
- **Delivery Risk:** Ability to deliver on time.
    *   Example Indicators: Lead time, delivery variance, production backlog.
- **Financial Risk:** Stability and solvency.
    *   Example Indicators: Credit score, late payments, financial health rating.
- **Technical Risk:** Capability and compliance.
    *   Example Indicators: Technical deviations, missing specs, low compliance score.
- **Quality Risk:** Historical performance.
    *   Example Indicators: NCR rate, defect returns, quality certification gaps.
- **HSE / Compliance Risk:** Safety and certification.
    *   Example Indicators: Missing HSE docs, expired ISO certifications.
- **Geopolitical / Supply Risk:** Location and route exposure.
    *   Example Indicators: Country risk, route congestion, sanctions.
- **ESG / Reputation Risk:** Sustainability and ethics.
    *   Example Indicators: ESG score, media exposure, carbon profile.
"""

    formatted_weights = "\n".join(
        [f"- {cat}: {weight}%" for cat, weight in risk_weights.items() if weight > 0]
    )

    template = """You are an expert in procurement risk analysis.
Analyze the provided offer text and generate a comprehensive risk analysis in JSON format.

**Offer Text:**
{offer_text}

{risk_dimensions_instructions}

**User-Provided Risk Dimension Weights:**
{formatted_weights}

**Risk Scoring Logic:**
1. Score each dimension (1-5)
2. Apply weights to calculate weighted score
3. Compute overall score (1-5)
4. Translate to Risk Level: 1.0-2.0: Low, 2.1-3.5: Medium, 3.6-5.0: High

**Output Format:**
Return a single JSON object with a "risk" key containing:
- "total_risk_score": A numerical risk score (1-5)
- "risk_level": "Low", "Medium", or "High"
- "summary": Brief narrative summary of key risks
- "dimension_scores": Individual scores for each dimension
- "detailed_risk_analysis": Table with headers and rows

**Example Output:**
{{
  "risk": {{
    "total_risk_score": 1.9,
    "risk_level": "Low",
    "summary": "Supplier shows balanced, low risk across all dimensions.",
    "dimension_scores": {{
      "Delivery Risk": 2.0,
      "Financial Risk": 2.5,
      "Technical Risk": 1.5
    }},
    "detailed_risk_analysis": {{
      "headers": ["Risk Factor", "Score", "Supporting Evidence", "Mitigation Recommendations"],
      "rows": [
        ["Delivery Risk", "2.0", "Lead time of 4 weeks, within acceptable range.", "None needed."]
      ]
    }}
  }}
}}

Begin!

**Analysis:**
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["offer_text"],
        partial_variables={
            "risk_dimensions_instructions": risk_dimensions_instructions,
            "formatted_weights": formatted_weights,
        },
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def create_scoring_agent(weights):
    """
    Creates an agent that scores an offer based on analysis data and weights.
    """
    llm = _get_llm()

    formatted_weights = "\n".join(
        [f"- {cat}: {weight}%" for cat, weight in weights.items() if weight > 0]
    )

    template = """You are a procurement scoring specialist.
Given the analysis of an offer and user-defined weights, you must calculate scores.

**Analysis Data (JSON):**
{analysis_data}

**User-Provided Weights:**
{formatted_weights}

**Scoring Instructions:**
1. Calculate percentage scores (0-100) for: Price, Delivery, Risk, and Compliance
2. Calculate `total_weighted_score` = sum of (category_score * weight)
   Example: Price (80, 50%) + Delivery (90, 50%) = (80*0.5) + (90*0.5) = 85

**Output Format:**
{{
  "total_weighted_score": 85,
  "category_scores": {{
    "Price": 80,
    "Delivery": 90,
    "Risk": 88,
    "Compliance": 95
  }}
}}

Begin!

**Scoring:**
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["analysis_data"],
        partial_variables={"formatted_weights": formatted_weights},
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def create_final_output_agent():
    """
    Creates an agent that combines all analysis pieces into the final JSON output.
    """
    llm = _get_llm()

    json_format_instructions = """
{
  "offer_name": "Name of the offer",
  "supplier_name": "Name of the supplier",
  "summary_metrics": {
    "Total Price": "...",
    "Lead Time": "...",
    "Payment Terms": "..."
  },
  "risk": {
    "total_risk_score": 1.9,
    "risk_level": "Low/Medium/High",
    "summary": "Brief risk summary",
    "dimension_scores": {},
    "detailed_risk_analysis": {
      "headers": [],
      "rows": []
    }
  },
  "recommendation": "Best Offer/Good Alternative/Not Recommended",
  "total_weighted_score": 85,
  "category_scores": {
    "Price": 80,
    "Delivery": 90,
    "Risk": 88,
    "Compliance": 95
  },
  "detailed_gap_analysis": {
    "headers": [],
    "rows": []
  }
}
"""

    template = """You are the final assembly agent.
Combine outputs from gap analysis, risk analysis, and scoring agents into a single JSON object.

**Offer Name:** {offer_name}
**Supplier Name:** {supplier_name}
**Gap Analysis Output (JSON):** {gap_analysis_output}
**Risk Analysis Output (JSON):** {risk_analysis_output}
**Scoring Output (JSON):** {scoring_output}

**Summary Metrics:**
Extract "Total Price", "Lead Time", and "Payment Terms" from gap analysis.

**Recommendation:**
Based on `total_weighted_score`:
- Score > 90: "Best Offer"
- Score 80-90: "Good Alternative"
- Score < 80: "Not Recommended"

**Final JSON Structure:**
{escaped_json_instructions}

Combine all provided information into a single JSON object following the structure precisely.

Begin!

**Final JSON Object:**
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "offer_name",
            "supplier_name",
            "gap_analysis_output",
            "risk_analysis_output",
            "scoring_output",
        ],
        partial_variables={
            "escaped_json_instructions": json_format_instructions
        },
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def create_comparison_agent():
    """
    Creates an agent that generates a comparison table, AI insights, and action list.
    """
    llm = _get_llm()

    template = """You are an expert procurement analyst.
Given analyzed procurement offers in JSON format, generate:
1. Offer Comparison Table
2. AI Highlights & Insights
3. Action List

**Input Analysis Data (JSON Array):**
{analyzed_offers_json}

**Output Format:**
```json
{
  "comparison_summary": {
    "comparison_table": [
      {
        "criterion": "Price (USD)",
        "Supplier A": "$1.05M",
        "Supplier B": "$1.08M",
        "observation": "B +5%, C -2%",
        "highlight": "yellow"
      }
    ],
    "ai_insights": "Supplier B's delivery time is twice as long...",
    "action_list": [
      {
        "action": "Ask Supplier B to explain 20-week delivery time",
        "responsible": "Buyer",
        "status": "Open",
        "due_date": "2025-11-30"
      }
    ]
  }
}
```

Begin!

**Comparison Summary:**
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["analyzed_offers_json"],
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def _read_and_combine_offer_files(offer_files_group, retriever):
    """
    Reads and combines the content of files for a single offer.
    """
    llm = _get_llm()
    tools = file_reading_tools + [currency_converter, calculator_tool]

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=180
    )

    combined_content = ""
    supplier_name = "Unknown Supplier"

    for file_path in offer_files_group:
        custom_instructions = f"""
Your purpose is to read the file at the following path and extract its content and the supplier's name.
If you find any costs in a currency other than USD, you MUST use the `currency_converter` tool to convert it to USD.
File to read: {file_path}

Return a JSON object containing:
- "content": The full text content of the file.
- "supplier_name": The extracted name of the supplier.
"""

        input_question = f"{custom_instructions}\n\nQuestion: Read the file and extract the content and supplier name."

        try:
            result = agent_executor.invoke({"input": input_question})

            json_start = result["output"].find("{")
            json_end = result["output"].rfind("}") + 1
            if json_start != -1 and json_end != -1:
                data = json.loads(result["output"][json_start:json_end])
                content = data.get("content", "")
                combined_content += content + "\n\n"
                if supplier_name == "Unknown Supplier" and data.get("supplier_name"):
                    supplier_name = data.get("supplier_name")

                # Add the content to the retriever
                retriever.add_to_retriever(
                    [
                        Document(
                            page_content=content,
                            metadata={"supplier": supplier_name, "file": file_path}
                        )
                    ]
                )
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            traceback.print_exc()
            continue

    return {"content": combined_content, "supplier_name": supplier_name}


def create_analysis_agent(
    weights=None, risk_weights=None, num_offers=0, offer_files=None
):
    """
    Orchestrates the analysis of procurement offers using specialized agents.

    Args:
        weights: Dictionary of category weights for scoring
        risk_weights: Dictionary of risk dimension weights
        num_offers: Number of offers to analyze
        offer_files: List of lists, where each inner list contains file paths for one offer

    Returns:
        Dictionary with 'output', 'comparison_summary', and 'retriever'
    """
    retriever = Retriever()

    # Default weights if not provided
    if weights is None:
        weights = {
            "Total Cost of Ownership (TCO)": 25,
            "Payment Terms": 10,
            "Price Stability": 5,
            "Lead Time": 20,
            "Technical Specifications": 25,
            "Certifications": 5,
            "Incoterms": 5,
            "Warranty": 5,
        }

    if risk_weights is None:
        risk_weights = {
            "Delivery Risk": 15,
            "Financial Risk": 15,
            "Technical Risk": 15,
            "Quality Risk": 15,
            "HSE / Compliance Risk": 15,
            "Geopolitical / Supply Risk": 10,
            "ESG / Reputation Risk": 15,
        }

    # Initialize Specialized Agents
    gap_analysis_agent = create_gap_analysis_agent()
    risk_analysis_agent = create_risk_analysis_agent(risk_weights=risk_weights)
    scoring_agent = create_scoring_agent(weights=weights)
    final_output_agent = create_final_output_agent()
    comparison_agent = create_comparison_agent()

    # Orchestration Logic
    final_json_outputs = []

    for i, offer_files_group in enumerate(offer_files):
        offer_name = f"Offer {i + 1}"
        print(f"\n{'='*60}")
        print(f"Processing {offer_name}: {offer_files_group}")
        print(f"{'='*60}\n")

        # Read and combine offer files
        offer_data = _read_and_combine_offer_files(offer_files_group, retriever)
        offer_text = offer_data.get("content")
        supplier_name = offer_data.get("supplier_name", "Unknown Supplier")

        if not offer_text:
            print(f"Warning: No content extracted from {offer_name}")
            continue

        print(f"Extracted supplier: {supplier_name}")
        print(f"Content length: {len(offer_text)} characters")

        # Gap Analysis
        print(f"\n--- Running Gap Analysis for {offer_name} ---")
        gap_result = gap_analysis_agent.invoke({"offer_text": offer_text})
        gap_analysis_output = _clean_json_string(gap_result.get("text", "{}"))
        print(f"Gap Analysis Output: {gap_analysis_output[:200]}...")

        # Risk Analysis
        print(f"\n--- Running Risk Analysis for {offer_name} ---")
        risk_result = risk_analysis_agent.invoke({"offer_text": offer_text})
        risk_analysis_output = _clean_json_string(risk_result.get("text", "{}"))
        print(f"Risk Analysis Output: {risk_analysis_output[:200]}...")

        # Combine analyses
        try:
            gap_json = json.loads(gap_analysis_output)
            risk_json = json.loads(risk_analysis_output)
            combined_analysis = {**gap_json, **risk_json}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {offer_name}: {e}")
            traceback.print_exc()
            continue

        # Scoring
        print(f"\n--- Running Scoring for {offer_name} ---")
        scoring_result = scoring_agent.invoke(
            {"analysis_data": json.dumps(combined_analysis)}
        )
        scoring_output = _clean_json_string(scoring_result.get("text", "{}"))
        print(f"Scoring Output: {scoring_output}")

        # Final Assembly
        print(f"\n--- Assembling Final Output for {offer_name} ---")
        final_result = final_output_agent.invoke(
            {
                "offer_name": offer_name,
                "supplier_name": supplier_name,
                "gap_analysis_output": gap_analysis_output,
                "risk_analysis_output": risk_analysis_output,
                "scoring_output": scoring_output,
            }
        )
        final_offer_json_str = _clean_json_string(final_result.get("text", "{}"))

        try:
            final_offer_json = json.loads(final_offer_json_str)
            final_json_outputs.append(final_offer_json)

            # Add to retriever
            retriever.add_to_retriever(
                [
                    Document(
                        page_content=final_offer_json_str,
                        metadata={"supplier": supplier_name, "type": "analysis"},
                    )
                ]
            )
            print(f"✓ Successfully processed {offer_name}")
        except json.JSONDecodeError as e:
            print(f"Error parsing final JSON for {offer_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Processed {len(final_json_outputs)} offers successfully")
    print(f"{'='*60}\n")

    # Generate comparison summary
    comparison_summary = "{}"
    if final_json_outputs:
        print("--- Generating Comparison Summary ---")
        comparison_result = comparison_agent.invoke(
            {"analyzed_offers_json": json.dumps(final_json_outputs)}
        )
        comparison_summary = _clean_json_string(comparison_result.get("text", "{}"))
        print(f"Comparison Summary generated: {comparison_summary[:200]}...")

    return {
        "output": json.dumps(final_json_outputs),
        "comparison_summary": comparison_summary,
        "retriever": retriever,
    }


def create_chat_agent(analysis_json, retriever=None):
    """
    Creates a conversational agent for answering questions based on the analysis.
    """
    llm = _get_llm()
    llm.temperature = 0.3  # Slightly higher for conversation

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # Use retriever if available
    if retriever and retriever.get_retriever():
        print("Using retriever-based chat agent")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever.get_retriever(),
            return_source_documents=True,
        )

        # Wrap in conversation chain
        chat_agent = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True,
            prompt=PromptTemplate(
                template="""You are a procurement analyst assistant.
Use the retrieved context and your knowledge to answer questions clearly.

Context from retrieval: {context}

{chat_history}
Human: {question}
AI:""",
                input_variables=["question", "chat_history", "context"],
            ),
            input_key="question",
            output_key="answer",
        )
    else:
        print("Using standard chat agent (no retriever)")
        template = """You are a procurement analyst.
Your goal is to assist users in understanding the procurement offers and the analysis results.
You should answer questions in a clear, conversational, and human-like manner.

You have been provided with the following summary analysis:
{analysis_json}

{chat_history}
Human: {question}
AI:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "chat_history"],
            partial_variables={"analysis_json": str(analysis_json)},
        )

        chat_agent = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True,
            prompt=prompt,
            input_key="question",
            output_key="answer"
        )

    return chat_agent