"""
Blog Generator API using Llama 3.2 3B
Generates blog posts in Markdown based on company context and tone settings
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import httpx
import logging
from pathlib import Path
import PyPDF2
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Blog Generator API",
    description="Generate blog posts using LLM with configurable tone",
    version="1.0.0"
)

# Global state 
app_state = {
    "tone": "professional",  # Default tone
    "example_post": "",
    "company_description": ""
}

# Pydantic models for request/response validation
class ToneRequest(BaseModel):
    tone: str = Field(..., description="Tone of voice: concise, playful, authoritative, professional, etc.")

class ToneResponse(BaseModel):
    tone: str

class GenerateRequest(BaseModel):
    purpose: str = Field(..., description="Purpose of the blog post")
    language: str = Field(..., description="Language for the blog post (e.g., Dutch, English)")

class GenerateResponse(BaseModel):
    markdown: str

class HealthResponse(BaseModel):
    status: str
    model: str
    documents_loaded: bool

# PDF parsing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise

# Load documents at startup
@app.on_event("startup")
async def load_documents():
    """Load example post and company description from PDFs"""
    try:
        logger.info("Loading documents...")
        
        # Try to load from /app/documents (Docker path) or ./documents (local)
        base_paths = [Path("/app/documents"), Path("./documents"), Path(".")]
        
        for base_path in base_paths:
            example_path = base_path / "example_post.pdf"
            company_path = base_path / "company_description.pdf"
            
            if example_path.exists() and company_path.exists():
                app_state["example_post"] = extract_text_from_pdf(str(example_path))
                app_state["company_description"] = extract_text_from_pdf(str(company_path))
                logger.info(f"Documents loaded successfully from {base_path}")
                logger.info(f"Example post length: {len(app_state['example_post'])} chars")
                logger.info(f"Company description length: {len(app_state['company_description'])} chars")
                return
        
        raise FileNotFoundError("Could not find example_post.pdf and company_description.pdf")
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise

# Ollama interaction 
async def generate_with_ollama(prompt: str, temperature: float = 0.7) -> str:
    """Send prompt to Ollama and get response"""
    import os
    
    # Support both local and Docker environments
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {ollama_host}. Is it running?")
        raise HTTPException(status_code=503, detail="LLM service unavailable. Ensure Ollama is running.")
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

def build_prompt(purpose: str, language: str, tone: str) -> str:
    """Construct the prompt for blog generation"""
    
    # Map tone to descriptive guidance
    tone_descriptions = {
        "concise": "brief, to-the-point, no fluff",
        "playful": "friendly, engaging, uses metaphors and light humor",
        "authoritative": "confident, expert, data-driven",
        "professional": "polished, credible, balanced",
        "Inspirational": "motivational, encouraging, uplifting"
    }
    
    tone_desc = tone_descriptions.get(tone.lower(), tone)
    
    prompt = f"""You are a professional content writer for the company. Your task is to generate a blog post in Markdown format.

COMPANY CONTEXT:
{app_state['company_description'][:1500]}

STYLE REFERENCE (example post):
{app_state['example_post'][:1000]}

TASK:
Generate a blog post with the following specifications:
- Purpose: {purpose}
- Language: {language}
- Tone: {tone_desc}

REQUIREMENTS:
1. Start with a compelling title using # (H1 heading)
2. Include an "Introduction" section
3. Use ## for section headings
4. Include practical, actionable information
5. Use bullet points for lists (-, not *)
6. Keep paragraphs short (2-4 sentences)
7. Include numbered steps where appropriate
8. Match the approachable, practical style of the example post
9. End with a clear call-to-action or next steps
10. Write entirely in {language}

Generate ONLY the Markdown blog post, nothing else:"""

    return prompt

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import os
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        # Check if Ollama is accessible
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            ollama_status = response.status_code == 200
    except:
        ollama_status = False
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "model": "llama3.2:3b",
        "documents_loaded": bool(app_state["example_post"] and app_state["company_description"])
    }

@app.post("/settings/tone", response_model=ToneResponse)
async def set_tone(request: ToneRequest):
    """Set the global tone of voice for blog generation"""
    logger.info(f"Setting tone to: {request.tone}")
    app_state["tone"] = request.tone
    return {"tone": request.tone}

@app.post("/generate", response_model=GenerateResponse)
async def generate_blog(request: GenerateRequest):
    """Generate a blog post based on purpose and language"""
    
    # Validate documents are loaded
    if not app_state["example_post"] or not app_state["company_description"]:
        raise HTTPException(status_code=500, detail="Documents not loaded")
    
    logger.info(f"Generating blog post: purpose='{request.purpose[:50]}...', language={request.language}, tone={app_state['tone']}")
    
    start_time = time.time()
    
    # Build prompt with current tone
    prompt = build_prompt(
        purpose=request.purpose,
        language=request.language,
        tone=app_state["tone"]
    )
    
    # Generate content using Ollama
    markdown_content = await generate_with_ollama(prompt)
    
    elapsed = time.time() - start_time
    logger.info(f"Blog generated in {elapsed:.2f}s, length: {len(markdown_content)} chars")
    
    return {"markdown": markdown_content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)