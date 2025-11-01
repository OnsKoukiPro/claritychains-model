from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI as ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.chains.llm import LLMChain
from tools.file_tools import file_reading_tools
from tools.calculator_tool import calculator_tool
#from tools.data_normalizer_tool import normalize_offer_data


def create_analysis_agent(weights=None, num_offers=0, offer_files=None):
    """
    Creates a chain for generating a structured JSON analysis of offers.
    """
    llm = ChatOpenAI(
        azure_deployment="ClarityChain",  # or your deployment
        api_version="2024-08-01-preview",
        temperature=0,
    )

    tools = [
        calculator_tool,
        *file_reading_tools,
    ]

    # Format weights for the prompt
    formatted_weights = "No specific weights provided."
    if weights:
        formatted_weights = "\n".join(
            [f"- {cat}: {weight}%" for cat, weight in weights.items() if weight > 0]
        )

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

    json_format_instructions = """
    For each offer, provide a JSON object with the following structure. Do not add any text outside of the JSON array.
    {
      "offer_name": "Name of the offer (e.g., Offer 1)",
      "supplier_name": "Name of the supplier",
      "summary_metrics": {
        "Total Price": "...",
        "Lead Time": "...",
        "Payment Terms": "..."
      },
      "risk": {
        "level": "Low, Medium, or High",
        "score": "A score from 0 to 100",
        "summary": "A brief summary of the risk assessment."
      },
      "recommendation": "e.g., Best Offer, Good Alternative, Not Recommended",
      "total_weighted_score": "A single overall score from 0-100, calculated based on the weighted category scores.",
      "category_scores": {
          "Price": "A percentage score (0-100) for the price competitiveness.",
          "Delivery": "A percentage score for the lead time and delivery terms.",
          "Risk": "A percentage score representing the overall risk (e.g., 100 - assessed risk value).",
          "Compliance": "A percentage score for standards and technical compliance."
      },
      "detailed_gap_analysis": {
          "headers": ["An array of strings for the table headers, e.g., 'Category', 'Supplier Name', 'Gap Analysis'"],
          "rows": [
              ["An array of strings for the first row"],
              ["An array of strings for the second row"]
          ]
      },
      "detailed_risk_analysis": {
          "headers": ["An array of strings for the risk table headers, e.g., 'Risk Category', 'Assessment', 'Score'"],
          "rows": [
              ["An array of strings for the first risk row"],
              ["An array of strings for the second risk row"]
          ]
      }
    }
    Return a single, valid JSON array of these objects, one for each offer.
    """

    template = """You are an expert procurement analyst. Your task is to analyze the provided procurement documents and generate a comprehensive, structured JSON output.

You have been provided with the following offer files:
{offer_files}

**ANALYSIS REQUIREMENTS:**

1.  **Read Files:** You MUST use the file reading tools (such as read_docx, read_pdf, etc.) to read the content of each offer file.
2.  **Detailed Gap Analysis:** For the `detailed_gap_analysis` part of the JSON, you MUST create a table structure (headers and rows). The analysis MUST cover the following categories. For each category, you must fill in the values for each supplier and provide a comparative analysis. Do not leave any cells blank; use 'Not found' or 'N/A' if information is missing.
{detailed_category_instructions}

3.  **Risk Analysis:** Perform a detailed risk analysis for each offer.

4.  **Scoring:** Based on the user-provided weights below, calculate percentage scores (0-100) for the main categories for each offer: Price, Delivery, Risk, and Compliance. A higher score is better.

5.  **Recommendation:** Provide a final recommendation for each offer.
6.  **Total Weighted Score:** You MUST calculate a `total_weighted_score` for each offer. This score is the sum of each category score multiplied by its respective weight. For example, if Price (score=80, weight=50%) and Delivery (score=90, weight=50%), the total weighted score is (80 * 0.5) + (90 * 0.5) = 85.

**USER-PROVIDED WEIGHTS:**
{formatted_weights}

If the user has provided additional evaluation criteria in their question, you must also consider it.

Question: {question}

IMPORTANT: Your final output must be a single, valid JSON array. The JSON array must contain exactly {num_offers} objects, one for each of the {num_offers} offers presented. You will be penalized if you do not return all offers. Each object must follow these instructions precisely:
{escaped_json_instructions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. After each thought, you must provide an Action.
Action: The action to take, should be one of [{tool_names}]. The `Action:` line must contain *only* the tool name, without any extra characters, markdown, or formatting.
Action Input: the input to the action. The `Action Input:` line must contain *only* the input for the tool, and it must be on a single line.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Thought: I must use a tool.
{agent_scratchpad}

TOOLS:
------
{tools}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "input", "agent_scratchpad", "tools"],
        partial_variables={
            "offer_files": "\n".join(offer_files),
            "num_offers": num_offers,
            "formatted_weights": formatted_weights,
            "detailed_category_instructions": detailed_category_instructions,
            "escaped_json_instructions": json_format_instructions.replace(
                "{{", "{{{{"
            ).replace("}}", "}}}}"),
            "tool_names": ", ".join([t.name for t in tools]),
        },
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def create_chat_agent(analysis_json):
    """
    Creates a conversational agent for answering questions based on the analysis.
    """
    llm = ChatOpenAI(
        azure_deployment="ClarityChain",  # or your deployment
        api_version="2024-08-01-preview",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

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
        partial_variables={"analysis_json": analysis_json},
    )

    qa_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    chat_agent = ConversationChain(
        llm=qa_chain.llm, memory=memory, verbose=True, prompt=prompt, input_key="question", output_key="answer"
    )

    return chat_agent
