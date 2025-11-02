from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import json
from typing import List, Dict, Any
import markdown
import datetime
import traceback

from offer_agent import create_analysis_agent, create_chat_agent

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for staged offers and chains
staged_offers: List[List[str]] = []
chain_state: Dict[str, Any] = {}
current_run_dir: str = None


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "agent-api"}


@app.post("/api/add-offer")
async def add_offer_api(files: List[UploadFile] = File(...)):
    global staged_offers, chain_state, current_run_dir

    try:
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
    except Exception as e:
        print(f"ERROR in add_offer_api: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to add offer: {str(e)}")


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

    try:
        print(f"DEBUG: Starting analysis with {len(staged_offers)} staged offers")
        print(f"DEBUG: Staged offers content: {staged_offers}")

        if not staged_offers:
            raise HTTPException(
                status_code=400, detail="Please add at least one offer for analysis."
            )

        offer_files = [file for offer in staged_offers for file in offer]
        print(f"Number of staged offers: {len(staged_offers)}")
        print(f"Offer files to analyze: {offer_files}")

        # Verify files exist
        for file_path in offer_files:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"File not found: {file_path}. Please re-upload offers."
                )

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

        print("Creating analysis agent...")
        analysis_agent = create_analysis_agent(
            weights=weights, num_offers=len(staged_offers), offer_files=offer_files
        )

        print("Invoking agent...")
        result = analysis_agent.invoke(
            {"input": eval_criteria or "Analyze all offers"}
        )
        answer = result.get("output", "").strip()
        print(f"Agent output:\n{answer}")

        # Parse the JSON response
        analysis_json = {}
        json_start = answer.find("[")
        json_end = answer.rfind("]") + 1

        if json_start != -1 and json_end != -1:
            json_str = answer[json_start:json_end]

            # Clean up markdown code blocks if present
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            try:
                analysis_json = json.loads(json_str.strip())
                print(f"Successfully parsed JSON with {len(analysis_json)} offers")

                # Post-process to set the best offer
                if (
                    analysis_json
                    and isinstance(analysis_json, list)
                    and len(analysis_json) > 0
                ):
                    try:
                        # Find the offer with the highest score
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
                                offer["recommendation"] = "Good Alternative"
                    except (ValueError, TypeError) as e:
                        print(
                            f"Could not determine best offer due to score format error: {e}"
                        )

            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"Failed to parse JSON from:\n{json_str}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not parse JSON from AI response: {str(e)}"
                )
        else:
            print(f"No JSON found in output:\n{answer}")
            raise HTTPException(
                status_code=500,
                detail="AI did not return valid JSON. Please try again."
            )

        if isinstance(analysis_json, dict) and "error" in analysis_json:
            raise HTTPException(status_code=500, detail=analysis_json["error"])

        print("Creating chat agent...")
        chat_agent = create_chat_agent(analysis_json)

        chain_state = {"chat_agent": chat_agent, "analysis": analysis_json}

        number_of_offers = len(staged_offers)

        # Don't clear current_run_dir yet - files might still be needed
        # current_run_dir = None
        staged_offers = []

        return JSONResponse(
            content={
                "message": f"Successfully analyzed {number_of_offers} offers.",
                "analysis": analysis_json,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in analyze_offers_api: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}\n\nCheck server logs for details."
        )


@app.post("/api/chat")
async def chat_api(request: Dict[str, Any]):
    global chain_state

    try:
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

        chat_agent = chain_state["chat_agent"]

        result = chat_agent.invoke({"question": user_text})
        answer = result["answer"]
        return JSONResponse(content={"role": "assistant", "content": answer})

    except Exception as e:
        print(f"ERROR in chat_api: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={
                "role": "assistant",
                "content": f"Sorry, a critical error occurred: {str(e)}",
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