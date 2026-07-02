from openai import OpenAI
import json
from multimodal_rag.config import get_openai_api_key, OPENAI_CHAT_MODEL

class ResponseGenerator:
    """Generates context-aware grounded answers with source citations using OpenAI."""
    def __init__(self):
        self.api_key = get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def reload_client(self):
        """Reload API key and client from config."""
        self.api_key = get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> dict:
        """
        Generates a grounded response based on the hybrid-retrieved chunks.
        Returns a dict:
        {
           "answer": str,
           "sources": list[dict] # exact chunks used as context
        }
        """
        self.reload_client()
        
        if not self.api_key or not self.client:
            return {
                "answer": (
                    "**OpenAI API Key is missing or invalid.**\n\n"
                    "Please set your `OPENAI_API_KEY` in the Settings panel or in the environment "
                    "to generate LLM responses. "
                    "However, the hybrid retrieval mechanism is functional, and you can see the retrieved chunks below!"
                ),
                "sources": retrieved_chunks
            }

        if not retrieved_chunks:
            return {
                "answer": "No relevant document passages were found. Please ingest documents first or refine your search query.",
                "sources": []
            }

        # 1. Format the context block
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown")
            page_num = meta.get("page_number", "Unknown")
            section = meta.get("section_title", "Unknown")
            ctype = meta.get("type", "text")
            
            part = (
                f"=== CONTEXT CHUNK {idx + 1} ===\n"
                f"Document: {doc_name}\n"
                f"Page Number: {page_num}\n"
                f"Section Header: {section}\n"
                f"Content Type: {ctype.upper()}\n"
                f"Content:\n{chunk['content']}\n"
                f"=========================\n"
            )
            context_parts.append(part)
            
        context_str = "\n".join(context_parts)

        # 2. Build system and user prompts
        system_prompt = (
            "You are an expert Multimodal RAG (Retrieval-Augmented Generation) assistant.\n"
            "Your task is to answer the user's query based ONLY on the provided context chunks.\n"
            "The context chunks contain text paragraphs, markdown tables, and detailed description of images/figures.\n\n"
            
            "CRITICAL RULES:\n"
            "1. Ground all answers. Do not assume or extrapolate beyond the facts directly mentioned in the context.\n"
            "2. If the context does not contain enough information to answer the question, state that you cannot find the answer.\n"
            "3. Format your answer using clean Markdown (bold text, lists, and headers where appropriate).\n"
            "4. Source Attribution: You MUST cite your sources inline for every statement or fact you reference.\n"
            "   Use the following precise format for citations:\n"
            "   - For Text passages: `[Doc: <doc_name>, Page: <page_num>]`\n"
            "   - For Tables: `[Table: <doc_name>, Page: <page_num>]`\n"
            "   - For Diagrams/Images/Figures: `[Figure: <doc_name>, Page: <page_num>]`\n"
            "   Place these citations immediately after the facts they support (e.g. 'Company X revenue grew by 15% in 2023 [Table: report.pdf, Page: 4].').\n"
            "5. Never use generic citations like [1] or [Chunk 1]. Always use the [Doc/Table/Figure: filename, Page: X] format.\n"
            "6. Rendering Tables: If the user asks to show, display, or print a table, you MUST output the complete table in standard Markdown format (with | columns and --- headers) based on the table content in the context. Do NOT summarize or say you cannot show it. You have full access and ability to output tables.\n"
            "7. Code & Simulation Groundedness: If the user asks for code, programming scripts, simulations, or mathematical models to be generated, you MUST verify if such code, script, or mathematical equations exist in the provided context.\n"
            "   - If they do NOT exist in the context, you MUST explicitly state that the document does not contain any code, scripts, or mathematical models for this simulation.\n"
            "   - You may then offer to provide a hypothetical simulation based on general knowledge, but you MUST clearly label it as 'Hypothetical (Non-Grounded) Simulation' so the user is not misled into thinking it comes from the document. Do not cite pages from the document for fabricated code."
        )

        user_prompt = (
            f"Context Chunks:\n"
            f"{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"Helpful, Grounded Answer:"
        )

        # 3. Request completion from OpenAI
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # low temperature for factual RAG responses
            )
            if hasattr(response, 'usage') and response.usage:
                from multimodal_rag.config import TokenTracker
                TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": retrieved_chunks
            }
        except Exception as e:
            print(f"[Generator] OpenAI answer generation failed: {e}")
            return {
                "answer": f"**Error generating answer from OpenAI API:**\n\n`{str(e)}`",
                "sources": retrieved_chunks
            }

    def rewrite_query(self, query: str, history: list[dict]) -> str:
        """Rewrites a follow-up query to be standalone based on the chat history."""
        self.reload_client()
        if not self.client or not history:
            return query

        system_prompt = (
            "You are a conversational search assistant. Given the chat history and a follow-up query, "
            "determine if the follow-up query is dependent on the chat history.\n"
            "If it is, rewrite it into a standalone search query that contains all necessary context (such as subjects, "
            "table names, figure IDs, or document references mentioned earlier) and can be searched independently.\n"
            "If it is not dependent and is already standalone, return the original query exactly as is.\n"
            "Do NOT add any introductions, explanations, or quotes. Output ONLY the standalone query."
        )

        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]])
        user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Query: {query}\n\nStandalone Query:"

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            if hasattr(response, 'usage') and response.usage:
                from multimodal_rag.config import TokenTracker
                TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
            rewritten = response.choices[0].message.content.strip()
            print(f"[Generator] Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"[Generator] Query rewriting failed: {e}. Using original.")
            return query

    def decompose_query(self, query: str) -> list[str]:
        """Decomposes a query into 1 to 3 distinct sub-queries for broader search coverage."""
        self.reload_client()
        if not self.client:
            return [query]

        system_prompt = (
            "You are an information retrieval assistant. Your task is to analyze a search query "
            "and determine if it asks about multiple independent topics, sections, tables, or comparative questions.\n"
            "If it does, decompose it into 1 to 3 distinct, simpler sub-queries that can be used for searching.\n"
            "If the query is already simple, return a list containing only the original query.\n"
            "Your output MUST be a valid JSON array of strings (e.g. [\"sub-query 1\", \"sub-query 2\"]).\n"
            "Do not include any markdown formatting like ```json or explanations. Output ONLY the JSON array."
        )

        user_prompt = f"Query: {query}\n\nSub-queries (JSON array):"

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            if hasattr(response, 'usage') and response.usage:
                from multimodal_rag.config import TokenTracker
                TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
            content = response.choices[0].message.content.strip()
            # Clean potential markdown block formatting
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()
            
            sub_queries = json.loads(content)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                print(f"[Generator] Decomposed query: '{query}' -> {sub_queries}")
                return [str(q) for q in sub_queries]
            return [query]
        except Exception as e:
            print(f"[Generator] Query decomposition failed: {e}. Using original.")
            return [query]

    def classify_intent(self, query: str) -> dict:
        """Classify if the query semantically targets all images/figures or all tables."""
        self.reload_client()
        if not self.client:
            # Fallback to keyword check when offline
            query_lower = query.lower()
            wants_images = any(w in query_lower for w in ["every figure", "all figures", "each figure", "figures in", "every image", "all images", "summarize figures", "diagrams", "charts"])
            wants_tables = any(w in query_lower for w in ["every table", "all tables", "each table", "tables in", "summarize tables"])
            return {"wants_all_images": wants_images, "wants_all_tables": wants_tables}
            
        system_prompt = (
            "Analyze the user's query and classify if they are requesting a summary, list, or overview of "
            "all/every figure/image/diagram or all/every table in the document.\n"
            "Respond ONLY with a JSON object containing two boolean keys: 'wants_all_images' and 'wants_all_tables'.\n"
            "Do not include markdown formatting, json tags, or explanation."
        )
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            if hasattr(response, 'usage') and response.usage:
                from multimodal_rag.config import TokenTracker
                TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
                
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()
            res = json.loads(content)
            return {
                "wants_all_images": bool(res.get("wants_all_images", False)),
                "wants_all_tables": bool(res.get("wants_all_tables", False))
            }
        except Exception:
            # Fallback
            query_lower = query.lower()
            wants_images = any(w in query_lower for w in ["every figure", "all figures", "each figure", "figures in", "every image", "all images", "summarize figures", "diagrams", "charts"])
            wants_tables = any(w in query_lower for w in ["every table", "all tables", "each table", "tables in", "summarize tables"])
            return {"wants_all_images": wants_images, "wants_all_tables": wants_tables}
