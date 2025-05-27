import requests
import os
from typing import Any, Dict, List, Literal, Optional, Union
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks


class myCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank"
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        try:
            user_prompt = data["messages"][-1]["content"]
        except (AttributeError, IndexError):
            return "Invalid input: 'messages' missing or improperly formatted."
        try:
            # Call AIRS service to scan the user prompt
            airs_response = test_airs(user_prompt)

            if airs_response.status_code != 200:
                return f"airs call failed (HTTP {airs_response.status_code})."
            if airs_response.json().get("action","") == "block":
                return "Request blocked by security policy."
        except Exception as e:
            return f"Error calling AIRS {e}"

def test_airs(data):
  airs_response = requests.post(
    "<AIRS-API-URL>", 
    headers={
        "x-pan-token": os.environ.get("AIRS_APIKEY"), 
        "Content-Type": "application/json"
    },
    json={
        "metadata": {
            "ai_model": "Test AI model",
            "app_name": "Google AI",
            "app_user": "test-user-1"
        },
        "contents": [
            {
                "prompt": data
            }
        ],
        #"tr_id": "1234",
        "ai_profile": {
            "profile_name": os.environ.get("AIRS_PROFILE_NAME")
        }
    },
    timeout=5,
    verify=False
  )
  return airs_response