"""
Client for interacting with the AI Agent API
"""
import requests
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AgentClient:
    """Client for AI procurement analysis agent"""

    def __init__(self, base_url: Optional[str] = None):
        default_url = "http://agent-api:8000" if os.getenv('DOCKER_ENV') else "http://localhost:8000"
        self.base_url = base_url or os.getenv("AGENT_API_URL", default_url)
        self.session = requests.Session()
        logger.info(f"Agent client configured with base URL: {self.base_url}")

    def health_check(self) -> bool:
        """Check if agent API is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Agent API health check failed: {e}")
            return False

    def add_offer(self, files: List[str]) -> Dict[str, Any]:
        """
        Add an offer to the staged offers

        Args:
            files: List of file paths to upload

        Returns:
            Response from API with staged offers
        """
        try:
            files_to_upload = []
            for file_path in files:
                if os.path.exists(file_path):
                    files_to_upload.append(
                        ('files', (os.path.basename(file_path), open(file_path, 'rb')))
                    )

            response = self.session.post(
                f"{self.base_url}/api/add-offer",
                files=files_to_upload
            )

            # Close file handles
            for _, (_, file_handle) in files_to_upload:
                file_handle.close()

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to add offer: {e}")
            return {"error": str(e)}

    def analyze_offers(
        self,
        eval_criteria: str = "",
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze all staged offers

        Args:
            eval_criteria: Additional evaluation criteria
            weights: Custom weights for analysis factors

        Returns:
            Analysis results with rankings and recommendations
        """
        try:
            # Default weights
            default_weights = {
                "tco_weight": 25,
                "payment_terms_weight": 10,
                "price_stability_weight": 5,
                "lead_time_weight": 20,
                "tech_spec_weight": 25,
                "certifications_weight": 5,
                "incoterms_weight": 5,
                "warranty_weight": 5
            }

            if weights:
                default_weights.update(weights)

            data = {
                "eval_criteria": eval_criteria,
                **default_weights
            }

            response = self.session.post(
                f"{self.base_url}/api/analyze",
                data=data
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to analyze offers: {e}")
            return {"error": str(e)}

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Chat with the AI agent about the analysis

        Args:
            message: User question or comment

        Returns:
            Agent response
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={"message": message}
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to chat with agent: {e}")
            return {"error": str(e), "role": "assistant", "content": f"Error: {str(e)}"}


# Convenience function for quick access
def get_agent_client() -> AgentClient:
    """Get a configured agent client instance"""
    return AgentClient()