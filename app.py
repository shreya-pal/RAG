# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64 # Not directly used in final app.py, but good to keep if image handling changes
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
KNOWLEDGE_BASE_DB_PATH = "knowledge_base.db" # Renamed from DB_PATH
SIMILARITY_THRESHOLD = 0.68
MAX_RETRIEVED_RESULTS = 10 # Total number of top similar chunks to retrieve initially
MAX_CONTEXT_CHUNKS_PER_SOURCE = 4 # Max number of chunks to use from a single document/post for LLM context

load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY") # Renamed from API_KEY, Get API key from environment variable

# Pydantic Models (Modified for promptfoo.yaml compatibility)
class QueryRequest(BaseModel):
    """Represents the request body for a RAG query."""
    question: str
    image: Optional[str] = None # Reverted from image_base64 to image for promptfoo compatibility

class SourceLink(BaseModel): # Renamed from LinkInfo
    """Represents a source link with URL and associated text snippet."""
    url: str
    text: str # Reverted from text_snippet to text for promptfoo compatibility

class QueryResponse(BaseModel):
    """Represents the response structure for a RAG query."""
    answer: str
    links: List[SourceLink] # Reverted from sources to links for promptfoo compatibility

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set at startup
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set. The application will not function correctly for API calls.")

# --- Database Operations ---

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(KNOWLEDGE_BASE_DB_PATH)
        conn.row_factory = sqlite3.Row # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize database tables on application startup if they don't exist
# This is a fallback/convenience; preprocess.py should typically run first.
if not os.path.exists(KNOWLEDGE_BASE_DB_PATH):
    logger.warning(f"Database {KNOWLEDGE_BASE_DB_PATH} not found. Attempting to initialize tables for first use.")
    try:
        conn_init = sqlite3.connect(KNOWLEDGE_BASE_DB_PATH)
        c = conn_init.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
        ''')
        conn_init.commit()
        conn_init.close()
        logger.info("Database tables initialized successfully on startup.")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database tables on startup: {e}")
        # The app might still run but database queries will fail without valid tables

# --- Vector Operations ---

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    try:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        if np.all(vec1_np == 0) or np.all(vec2_np == 0):
            return 0.0

        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity calculation: {e}")
        logger.error(traceback.format_exc())
        return 0.0

async def get_openai_embedding(text: str, max_retries: int = 3) -> Optional[List[float]]: # Renamed from get_embedding
    """Gets an embedding for the given text from the OpenAI API via aipipe proxy."""
    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY environment variable not set."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # OpenAI's `text-embedding-3-small` model has an input token limit of 8192.
    # While this function takes character length, it's a good practice to be aware of the underlying token limit.
    # If `text` is very long, it should ideally be pre-chunked or handled by the calling function.
    
    retries = 0
    while retries < max_retries:
        try:
            # logger.debug(f"Getting embedding for text (length: {len(text)})") # Set to debug for less verbosity
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}", # Standard practice to use "Bearer" prefix
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        # logger.debug("Successfully received embedding.") # Set to debug for less verbosity
                        return result["data"][0]["embedding"]
                    elif response.status == 429: # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after {5 * (retries + 1)}s (retry {retries+1}/{max_retries}). Details: {error_text}")
                        await asyncio.sleep(5 * (retries + 1)) # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error from Embedding API (status {response.status}): {error_text}. Text snippet: '{text[:100]}...'"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except aiohttp.ClientError as e:
            error_msg = f"Network or client error during embedding API call (attempt {retries+1}/{max_retries}): {e}. Text snippet: '{text[:100]}...'"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)
        except asyncio.TimeoutError:
            error_msg = f"Embedding API request timed out (attempt {retries+1}/{max_retries}). Text snippet: '{text[:100]}...'"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)
        except Exception as e:
            error_msg = f"Unexpected error getting embedding (attempt {retries+1}/{max_retries}): {e}. Text snippet: '{text[:100]}...'"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)
    return None # Should not be reached if retries are exhausted and exception is raised

async def get_query_embedding(question: str, image_base64_data: Optional[str] = None) -> List[float]: # Renamed from process_multimodal_query
    """
    Generates an embedding for the query. If an image_base64_data is provided,
    it first uses a multimodal model (GPT-4o-mini) to describe the image in context of the question,
    then embeds the combined text.
    """
    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY environment variable not set. Cannot process multimodal queries."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    if not image_base64_data:
        logger.info("No image provided. Generating embedding for text-only query.")
        embedding = await get_openai_embedding(question)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for text query.")
        return embedding

    logger.info("Processing multimodal query with image.")
    try:
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        image_url_data = f"data:image/jpeg;base64,{image_base64_data}" # Assume JPEG for simplicity, can be dynamic
        
        payload = {
            "model": "gpt-4o-mini", # Use GPT-4o-mini for multimodal capabilities
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image in the context of the question: '{question}'. Describe what you see that is relevant to answering the question."},
                        {"type": "image_url", "image_url": {"url": image_url_data}}
                    ]
                }
            ],
            "max_tokens": 500 # Limit description length to avoid excessively long combined query
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=90) as response: # Increased timeout for Vision API
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description (length: {len(image_description)}).")
                    
                    # Combine the original question with the image description for embedding
                    combined_query = f"Question: {question}\nImage Context: {image_description}"
                    return await get_openai_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image with Vision API (status {response.status}): {error_text}")
                    logger.info("Falling back to text-only query due to Vision API error.")
                    return await get_openai_embedding(question) # Fallback to text-only
    except Exception as e:
        logger.error(f"Exception during multimodal query processing: {e}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to text-only query due to exception.")
        return await get_openai_embedding(question) # Fallback to text-only

# --- Retrieval & Context Building ---

async def retrieve_relevant_chunks(query_embedding: List[float], conn: sqlite3.Connection) -> List[Dict[str, Any]]: # Renamed from find_similar_content
    """
    Retrieves the most similar content chunks from the database based on query embedding.
    Groups results by source document/post and limits chunks per source to provide diverse context.
    """
    logger.info("Retrieving relevant chunks from database.")
    cursor = conn.cursor()
    all_raw_results = []

    # Define tables and their specific columns for consistent processing
    tables_config = [
        {"name": "discourse_chunks", "id_col": "post_id", "title_col": "topic_title", "url_col": "url", "type": "discourse"},
        {"name": "markdown_chunks", "id_col": "doc_title", "title_col": "doc_title", "url_col": "original_url", "type": "markdown"}
    ]

    for config in tables_config:
        table_name = config["name"]
        id_col = config["id_col"]
        title_col = config["title_col"]
        url_col = config["url_col"]
        source_type = config["type"]

        logger.debug(f"Querying {table_name} for relevant chunks.")
        cursor.execute(f"""
        SELECT id, {id_col} AS group_key, {title_col} AS title, chunk_index, content, embedding, {url_col} AS url_raw
        FROM {table_name}
        WHERE embedding IS NOT NULL
        """)
        
        chunks_from_table = cursor.fetchall()
        # logger.debug(f"Processing {len(chunks_from_table)} chunks from {table_name}.")

        for chunk_row in chunks_from_table:
            try:
                # Convert byte BLOB embedding back to Python list
                embedding = json.loads(chunk_row["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)

                if similarity >= SIMILARITY_THRESHOLD:
                    chunk_data = dict(chunk_row) # Convert sqlite3.Row to a mutable dictionary
                    chunk_data["source_type"] = source_type
                    chunk_data["similarity"] = float(similarity)

                    # Ensure URL is properly formatted/defaulted for each source type
                    url = chunk_data.pop("url_raw") # Get raw URL and remove from dict
                    if source_type == "markdown":
                        if not url or not url.startswith("http"):
                            # Fallback if original_url from markdown file is missing or malformed
                            url = f"https://docs.onlinedegree.iitm.ac.in/{chunk_data['title'].replace(' ', '-').lower()}"
                    elif source_type == "discourse" and not url.startswith("http"):
                        # This should ideally not happen if preprocess.py correctly stores full URL
                        logger.warning(f"Discourse URL malformed for chunk ID {chunk_data['id']}. Attempting auto-fix: {url}")
                        # If a Discourse URL somehow lacks http, this is a very basic attempt to fix.
                        # The full URL should be stored during preprocessing (e.g., from topic_slug, topic_id, post_number).
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}" # Assuming the URL in DB was partial slug/path

                    chunk_data["url"] = url # Add cleaned URL back
                    all_raw_results.append(chunk_data)

            except json.JSONDecodeError as jde:
                logger.warning(f"Skipping chunk {chunk_row['id']} from {table_name} due to invalid embedding JSON: {jde}")
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_row['id']} from {table_name}: {e}")

    # Sort all results by similarity (descending)
    all_raw_results.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(f"Found {len(all_raw_results)} results above similarity threshold.")

    # Group by source document/post and select top chunks per group
    grouped_and_limited_chunks = {} # Key: unique doc identifier (e.g., "discourse_POST_ID", "markdown_DOC_TITLE")
    
    for result in all_raw_results:
        # Create a unique key for the document/post
        group_identifier = f"{result['source_type']}_{result['group_key']}"
        
        if group_identifier not in grouped_and_limited_chunks:
            grouped_and_limited_chunks[group_identifier] = []
        
        # Add chunks up to MAX_CONTEXT_CHUNKS_PER_SOURCE for each unique document/post
        if len(grouped_and_limited_chunks[group_identifier]) < MAX_CONTEXT_CHUNKS_PER_SOURCE:
            grouped_and_limited_chunks[group_identifier].append(result)
    
    final_retrieved_chunks = []
    for group_id, chunks in grouped_and_limited_chunks.items():
        # Ensure chunks within each group are sorted by similarity (they should be already but double-check)
        chunks.sort(key=lambda x: x["similarity"], reverse=True)
        final_retrieved_chunks.extend(chunks)

    # Finally, sort all selected chunks by similarity one last time to prioritize for LLM context building
    final_retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Limit the total number of chunks returned to MAX_RETRIEVED_RESULTS
    final_retrieved_chunks = final_retrieved_chunks[:MAX_RETRIEVED_RESULTS]
    logger.info(f"Returning {len(final_retrieved_chunks)} final relevant chunks after grouping and limiting.")
    return final_retrieved_chunks

async def enrich_chunks_with_neighbors(conn: sqlite3.Connection, relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # Renamed from enrich_with_adjacent_chunks
    """
    Enriches the content of retrieved chunks by adding adjacent chunks (previous and next)
    from the same document/post to provide more contextual information to the LLM.
    """
    logger.info(f"Attempting to enrich {len(relevant_chunks)} chunks with neighboring context.")
    cursor = conn.cursor()
    enriched_chunks = []

    for chunk in relevant_chunks:
        original_chunk_content = chunk["content"]
        enriched_content = original_chunk_content # Start with the chunk's own content
        source_type = chunk["source_type"]
        current_chunk_index = chunk["chunk_index"]

        # Determine how to query for neighbors based on source type
        if source_type == "discourse":
            item_id_column = "post_id"
            table_name = "discourse_chunks"
        elif source_type == "markdown":
            item_id_column = "doc_title" # For markdown, doc_title is used to group related chunks
            table_name = "markdown_chunks"
        else:
            enriched_chunks.append(chunk) # Unrecognized source type, add as is
            logger.warning(f"Skipping enrichment for chunk {chunk.get('id')} due to unknown source type: {source_type}.")
            continue

        item_id_value = chunk.get(item_id_column)
        if item_id_value is None:
            enriched_chunks.append(chunk) # Missing ID for lookup, add as is
            logger.warning(f"Skipping enrichment for chunk {chunk.get('id')} due to missing '{item_id_column}'.")
            continue

        # Get previous chunk
        prev_chunk_content = None
        if current_chunk_index > 0: # Only try to get previous if not the first chunk
            cursor.execute(f"""
            SELECT content FROM {table_name}
            WHERE {item_id_column} = ? AND chunk_index = ?
            """, (item_id_value, current_chunk_index - 1))
            prev_row = cursor.fetchone()
            if prev_row:
                prev_chunk_content = prev_row["content"]

        # Get next chunk
        next_chunk_content = None
        cursor.execute(f"""
        SELECT content FROM {table_name}
        WHERE {item_id_column} = ? AND chunk_index = ?
        """, (item_id_value, current_chunk_index + 1))
        next_row = cursor.fetchone()
        if next_row:
            next_chunk_content = next_row["content"]

        # Assemble enriched content, placing neighbors before and after the central chunk
        if prev_chunk_content:
            enriched_content = f"{prev_chunk_content}\n\n{enriched_content}"
        if next_chunk_content:
            enriched_content = f"{enriched_content}\n\n{next_chunk_content}"
        
        # Update the content of the chunk dictionary with the enriched content
        chunk["content"] = enriched_content
        enriched_chunks.append(chunk)

    logger.info(f"Finished enriching {len(enriched_chunks)} chunks.")
    return enriched_chunks

# --- LLM Interaction ---

async def generate_llm_answer(question: str, relevant_chunks: List[Dict[str, Any]], max_retries: int = 2) -> str: # Renamed from generate_answer
    """Generates an answer using the LLM based on the provided question and relevant context."""
    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY environment variable not set. Cannot generate LLM answers."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Sending question to LLM: '{question[:70]}...'")
            context_string = ""
            # Max 1500 chars per chunk is a simple way to stay within LLM context window.
            # A more robust solution for large contexts would involve token-aware trimming using libraries like tiktoken.
            for i, chunk in enumerate(relevant_chunks):
                source_type_display = "Discourse Post" if chunk["source_type"] == "discourse" else "Documentation"
                context_string += f"\n\n--- Source {i+1} ({source_type_display}) ---\nURL: {chunk['url']}\nContent: {chunk['content'][:1500]}"
            
            prompt_template = f"""Answer the following question comprehensively and concisely, based ONLY on the provided context.
            If you cannot answer the question based on the context, state "I don't have enough information to answer this question based on the provided context."
            
            Context:
            {context_string}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer.
            2. A "Sources:" section that lists the URLs and a brief relevant text snippet you used to answer.
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description of text relevant to the answer from source 1]
            2. URL: [exact_url_2], Text: [brief quote or description of text relevant to the answer from source 2]
            
            Make sure the URLs are copied exactly from the context without any changes.
            Ensure the text snippet is a direct, brief quote from the cited source content that directly supports the answer.
            """
            
            # logger.debug(f"Full prompt sent to LLM:\n{prompt_template}") # Use debug for full prompt content
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini", # Using a cost-effective yet capable model
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include exact URLs and brief relevant text snippets in your 'Sources:' section."},
                    {"role": "user", "content": prompt_template}
                ],
                "temperature": 0.3, # Lower temperature for more deterministic and factual outputs
                "max_tokens": 1000 # Limit the response length from LLM to control costs and ensure conciseness
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=90) as response: # Increased timeout for LLM response
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM.")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429: # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"LLM Rate limit reached, retrying after {3 * (retries + 1)}s (retry {retries+1}/{max_retries}). Details: {error_text}")
                        await asyncio.sleep(3 * (retries + 1)) # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer from LLM (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except aiohttp.ClientError as e:
            error_msg = f"Network or client error during LLM API call (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2 * retries)
        except asyncio.TimeoutError:
            error_msg = f"LLM API request timed out (attempt {retries+1}/{max_retries})."
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2 * retries)
        except Exception as e:
            error_msg = f"Unexpected error generating answer from LLM (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2 * retries)
    return "I am unable to generate an answer at this time due to persistent errors." # Fallback after exhausted retries

def parse_llm_response_and_sources(response_text: str, relevant_chunks: List[Dict[str, Any]]) -> QueryResponse: # Renamed from parse_llm_response
    """
    Parses the LLM's raw response string to extract the answer and structured source links.
    Includes a robust fallback to generate sources from the initially retrieved `relevant_chunks`
    if the LLM fails to provide them in the expected format.
    """
    logger.info("Parsing LLM response for answer and sources.")
    answer = ""
    parsed_sources: List[SourceLink] = []

    # First, try to split the response into answer and sources sections using common headings
    parts = re.split(r'Sources:|Source:|References:|Reference:', response_text, 1, re.IGNORECASE)

    answer = parts[0].strip()

    if len(parts) > 1:
        sources_section_text = parts[1].strip()
        # Split by newlines, then process each line
        source_lines = sources_section_text.split("\n")

        for line in source_lines:
            line = line.strip()
            if not line:
                continue

            # Remove common list markers (e.g., "1.", "-", "*") from the beginning of the line
            line = re.sub(r'^\s*(\d+\.|\*|-)\s*', '', line)

            url = ""
            text_snippet = ""

            # More flexible regex to find URL and Text:
            # Pattern 1: URL: [http://...] , Text: [Text snippet] (or similar with quotes)
            match_pattern1 = re.search(r'URL:\s*\[(.*?)\]\s*,\s*Text:\s*["\']?(.*?)["\']?$', line, re.IGNORECASE)
            if match_pattern1:
                url = match_pattern1.group(1).strip()
                text_snippet = match_pattern1.group(2).strip()
            else:
                # Pattern 2: URL: http://... , Text: Text snippet (no brackets, maybe just spaces)
                match_pattern2 = re.search(r'URL:\s*(\S+?)(?:\s*,\s*Text:\s*["\']?(.*?)["\']?)?$', line, re.IGNORECASE)
                if match_pattern2:
                    url = match_pattern2.group(1).strip()
                    if match_pattern2.group(2): # Check if the second group (text) was matched
                        text_snippet = match_pattern2.group(2).strip()
            
            # If a URL is found but the text snippet is empty, try to extract a generic snippet or the rest of the line
            if url and not text_snippet:
                # Attempt to remove the URL part and use the rest of the line as text_snippet
                remaining_line = re.sub(r'URL:\s*(\S+)', '', line, flags=re.IGNORECASE).strip().replace(',', '')
                if remaining_line:
                    text_snippet = remaining_line
                else:
                    text_snippet = "Referenced content" # Default snippet if no text found

            # Only add to sources if we have a valid URL (starts with http/https)
            if url and url.startswith("http"):
                parsed_sources.append(SourceLink(url=url, text=text_snippet)) # Reverted text_snippet to text

    # Fallback Mechanism: If LLM failed to parse or provide sources, generate from relevant chunks
    if not parsed_sources and relevant_chunks:
        logger.warning("No sources extracted from LLM response. Falling back to generating sources from relevant chunks.")
        unique_urls_for_fallback = set()
        for chunk in relevant_chunks:
            chunk_url = chunk.get('url')
            if chunk_url and chunk_url.startswith("http") and chunk_url not in unique_urls_for_fallback:
                # Use a small part of the content as the snippet for fallback
                snippet_text = chunk['content'][:150].replace('\n', ' ') # Take first 150 chars, normalize newlines
                if len(chunk['content']) > 150:
                    snippet_text += "..." # Add ellipsis if content was truncated
                parsed_sources.append(SourceLink(url=chunk_url, text=snippet_text)) # Reverted text_snippet to text
                unique_urls_for_fallback.add(chunk_url)

    logger.info(f"Parsed answer (length: {len(answer)}) and {len(parsed_sources)} sources.")
    return QueryResponse(answer=answer, links=parsed_sources) # Reverted sources to links

# --- API Endpoints ---

@app.post("/query", response_model=QueryResponse) # Explicitly define response model
async def query_knowledge_base(request: QueryRequest):
    """
    Handles RAG queries. Takes a question and optional image, retrieves relevant context,
    generates an answer using an LLM, and provides sources.
    """
    logger.info(f"Received query request: question='{request.question[:70]}...', image_provided={request.image is not None}")

    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY environment variable not set. Cannot process API calls."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    db_connection = None # Initialize conn to None for finally block
    try:
        db_connection = get_db_connection()
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Could not establish database connection.")
        
        # 1. Generate query embedding (handles multimodal input or text-only)
        query_embedding = await get_query_embedding(
            request.question,
            request.image # Reverted from image_base64 to image
        )
        if query_embedding is None:
            logger.error("Failed to generate query embedding. Cannot proceed with retrieval.")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # 2. Retrieve relevant chunks from the knowledge base
        relevant_chunks = await retrieve_relevant_chunks(query_embedding, db_connection)

        if not relevant_chunks:
            logger.info("No relevant information found in the knowledge base for the given query.")
            return QueryResponse(
                answer="I couldn't find any relevant information in my knowledge base related to your question. Please try rephrasing or provide more details.",
                links=[] # Reverted from sources to links
            )
        
        # 3. Enrich the retrieved chunks with adjacent neighbors for better context
        enriched_chunks = await enrich_chunks_with_neighbors(db_connection, relevant_chunks)
        
        # 4. Generate the answer using the LLM based on the question and enriched context
        llm_raw_response_text = await generate_llm_answer(request.question, enriched_chunks)
        
        # 5. Parse the LLM's response to extract the answer and structured sources
        final_response = parse_llm_response_and_sources(llm_raw_response_text, relevant_chunks)
        
        logger.info(f"Query processed successfully. Answer length: {len(final_response.answer)}, Sources found: {len(final_response.links)}") # Reverted sources to links
        return final_response
    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI HTTP exceptions directly to be handled by FastAPI
    except Exception as e:
        error_msg = f"An unexpected error occurred during query processing: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc()) # Log full traceback for debugging
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if db_connection:
            db_connection.close() # Ensure database connection is closed

# Health check endpoint for monitoring application status
@app.get("/health")
async def health_check():
    """Provides a health check for the API and its underlying data/dependencies."""
    db_connection = None
    try:
        db_connection = get_db_connection()
        if db_connection is None:
            raise Exception("Database connection failed during health check.")
        
        cursor = db_connection.cursor()
        
        # Check counts for discourse chunks
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_chunk_total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_chunk_embedded = cursor.fetchone()[0]
        
        # Check counts for markdown chunks
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_chunk_total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_chunk_embedded = cursor.fetchone()[0]
        
        return {
            "status": "healthy",
            "database_status": "connected",
            "openai_api_key_set": bool(OPENAI_API_KEY),
            "data_summary": {
                "discourse_chunks_total": discourse_chunk_total,
                "discourse_chunks_embedded": discourse_chunk_embedded,
                "markdown_chunks_total": markdown_chunk_total,
                "markdown_chunks_embedded": markdown_chunk_embedded,
                "total_chunks_ready_for_rag": discourse_chunk_embedded + markdown_chunk_embedded
            },
            "retrieval_parameters": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_retrieved_results": MAX_RETRIEVED_RESULTS,
                "max_context_chunks_per_source": MAX_CONTEXT_CHUNKS_PER_SOURCE
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "openai_api_key_set": bool(OPENAI_API_KEY)
            }
        )
    finally:
        if db_connection:
            db_connection.close() # Ensure database connection is closed

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # reload=True is useful for development as it restarts the server on code changes
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
