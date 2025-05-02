import google.generativeai as genai
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Consider making model name configurable
GEMINI_MODEL_NAME = "gemini-2.0-flash" 

def generate_answer_with_gemini(query: str, context_chunks: List[str], api_key: str) -> str | None:
    """Generates an answer to a query using Gemini, based on provided context.

    Args:
        query: The user's search query.
        context_chunks: A list of text chunks retrieved from the search.
        api_key: The Google AI API key.

    Returns:
        The generated answer string, or None if an error occurs.
    """
    if not api_key:
        logger.error("Gemini API key is missing. Cannot generate answer.")
        return "Error: Gemini API key not configured."
    if not context_chunks:
        logger.warning("No context provided to Gemini. Cannot generate answer.")
        # Return None, let the caller decide how to handle this (e.g., skip RAG)
        return None 

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Construct the prompt
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = (
            f"Based ONLY on the following context documents, please provide a concise answer to the query.\n"
            f"If the context does not contain the answer, state that the information is not available in the provided context.\n"
            f"Do not use any prior knowledge.\n\n"
            f"CONTEXT:\n"
            f"=======\n"
            f"{context_str}\n"
            f"=======\n\n"
            f"QUERY: {query}\n\n"
            f"ANSWER:"
        )

        logger.info(f"Sending query and {len(context_chunks)} context chunks to Gemini model: {GEMINI_MODEL_NAME}")
        # Increase safety settings slightly? Optional.
        # safety_settings = {
        #     'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
        # }
        response = model.generate_content(prompt) #, safety_settings=safety_settings)

        # Check for valid response text
        if response.parts:
            answer = "".join(part.text for part in response.parts)
            logger.info("Received response from Gemini.")
            return answer.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             logger.error(f"Gemini request blocked due to: {reason}")
             return f"Error: Content generation blocked by API safety filters ({reason})."
        else:
             # Handle cases where response might be empty without explicit blocking
             logger.error(f"Gemini returned an empty or unexpected response: {response}")
             return "Error: Received an empty or unexpected response from the API."

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        # Provide a more generic error to the user
        return f"Error: Failed to communicate with the Gemini API. Details: {e}" 