from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import json
from typing import List, Dict, Any
import markdown
import datetime

from offer_agent import create_analysis_agent, create_chat_agent

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for staged offers and chains
staged_offers: List[List[str]] = []
chain_state: Dict[str, Any] = {}
current_run_dir: str = None

# --- Helper functions ---
# Add this to agent/main.py after app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "agent-api"}

@app.post("/api/add-offer")
async def add_offer_api(files: List[UploadFile] = File(...)):
    global staged_offers, chain_state, current_run_dir

    if not current_run_dir:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_run_dir = os.path.join("runs", now)
        os.makedirs(current_run_dir, exist_ok=True)
        staged_offers = []
        chain_state = {}

    offer_files = []
    for file in files:
        file_path = os.path.join(current_run_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        offer_files.append(file_path)

    staged_offers.append(offer_files)

    return JSONResponse(
        content={
            "message": f"Offer with {len(offer_files)} file(s) added.",
            "staged_offers": staged_offers,
        }
    )


@app.post("/api/analyze")
async def analyze_offers_api(
    eval_criteria: str = Form(""),
    tco_weight: float = Form(25),
    payment_terms_weight: float = Form(10),
    price_stability_weight: float = Form(5),
    lead_time_weight: float = Form(20),
    tech_spec_weight: float = Form(25),
    certifications_weight: float = Form(5),
    incoterms_weight: float = Form(5),
    warranty_weight: float = Form(5),
):
    global staged_offers, chain_state, current_run_dir
    if not staged_offers:
        raise HTTPException(
            status_code=400, detail="Please add at least one offer for analysis."
        )

    offer_files = [file for offer in staged_offers for file in offer]
    print(f"Number of staged offers: {len(staged_offers)}")

    weights = {
        "Total Cost of Ownership (TCO)": tco_weight,
        "Payment Terms": payment_terms_weight,
        "Price Stability": price_stability_weight,
        "Lead Time": lead_time_weight,
        "Technical Specifications": tech_spec_weight,
        "Certifications": certifications_weight,
        "Incoterms": incoterms_weight,
        "Warranty": warranty_weight,
    }

    try:
        analysis_agent = create_analysis_agent(
            weights=weights, num_offers=len(staged_offers), offer_files=offer_files
        )

        result = analysis_agent.invoke(
            {"question": eval_criteria, "input": eval_criteria}
        )
        answer = result.get("output", "").strip()
        print(answer)

        # Parse the JSON response
        analysis_json = {}
        json_start = answer.find("[")
        json_end = answer.rfind("]") + 1
        if json_start != -1 and json_end != -1:
            json_str = answer[json_start:json_end]
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            try:
                analysis_json = json.loads(json_str.strip())
                # Post-process to set the best offer
                if (
                    analysis_json
                    and isinstance(analysis_json, list)
                    and len(analysis_json) > 0
                ):
                    try:
                        # Find the offer with the highest score, handling potential string values
                        best_offer = max(
                            analysis_json,
                            key=lambda x: float(
                                x.get("total_weighted_score", "0") or "0"
                            ),
                        )

                        # Set the "Best Offer" recommendation
                        for offer in analysis_json:
                            if offer is best_offer:
                                offer["recommendation"] = "Best Offer"
                            elif offer.get("recommendation") == "Best Offer":
                                # If agent recommended another as best, demote it.
                                offer["recommendation"] = "Good Alternative"
                    except (ValueError, TypeError) as e:
                        print(
                            f"Could not determine best offer due to score format error: {e}"
                        )

            except json.JSONDecodeError:
                print(f"Failed to parse JSON from raw output:\n{answer}")
                raise HTTPException(
                    status_code=500, detail="Could not parse JSON from AI response."
                )
            print(f"Number of offers in analysis_json: {len(analysis_json)}")
        else:
            print(f"Failed to parse JSON from raw output:\n{answer}")
            raise HTTPException(
                status_code=500, detail="Could not parse JSON from AI response."
            )

        if isinstance(analysis_json, dict) and "error" in analysis_json:
            raise HTTPException(status_code=500, detail=analysis_json["error"])

        chat_agent = create_chat_agent(analysis_json)

        chain_state = {"chat_agent": chat_agent, "analysis": analysis_json}

        current_run_dir = None
        number_of_offers = len(staged_offers)
        staged_offers = []

        return JSONResponse(
            content={
                "message": f"Successfully analyzed {number_of_offers} offers.",
                "analysis": analysis_json,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/api/chat")
async def chat_api(request: Dict[str, Any]):
    global chain_state
    user_text = request.get("message")
    if not user_text:
        return JSONResponse(
            content={"role": "assistant", "content": "Please enter a question."}
        )

    if not chain_state or "chat_agent" not in chain_state:
        return JSONResponse(
            content={
                "role": "assistant",
                "content": "Please upload and analyze documents before asking questions.",
            }
        )

    try:
        chat_agent = chain_state["chat_agent"]

        result = chat_agent.invoke({"question": user_text})
        answer = result["answer"]
        return JSONResponse(content={"role": "assistant", "content": answer})
    except Exception as e:
        return JSONResponse(
            content={
                "role": "assistant",
                "content": f"Sorry, a critical error occurred: {e}",
            }
        )


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


if __name__ == "__main__":
    import uvicorn

    if not os.path.exists("runs"):
        os.makedirs("runs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
