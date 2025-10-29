"""
Blog Generator API using Llama 3.2 3B
Generates blog posts in Markdown based on company context and tone settings
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import httpx
import logging
from pathlib import Path
import PyPDF2
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Blog Generator API",
    description="Generate blog posts using LLM with configurable tone and determinism controls",
    version="2.0.0"
)

# Configuration from environment variables
CONFIG = {
    "example_post_file": os.getenv("EXAMPLE_POST_FILE", "example_post.pdf"),
    "company_description_file": os.getenv("COMPANY_DESCRIPTION_FILE", "company_description.pdf"),
    "documents_dir": os.getenv("DOCUMENTS_DIR", "documents"),
    "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model_name": os.getenv("MODEL_NAME", "llama3.2:3b"),
    "default_tone": os.getenv("DEFAULT_TONE", "professional"),
    "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
    "max_purpose_length": int(os.getenv("MAX_PURPOSE_LENGTH", "500"))
}

# Global state 
app_state = {
    "tone": CONFIG["default_tone"],
    "example_post": "",
    "company_description": ""
}

# Pydantic models with validation

class ToneRequest(BaseModel):
    tone: str = Field(..., description="Tone of voice: concise, playful, authoritative, professional, etc.")
    
    @validator('tone')
    def validate_tone(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Tone cannot be empty")
        if len(v) > 50:
            raise ValueError("Tone must be 50 characters or less")
        return v.strip()

class ToneResponse(BaseModel):
    tone: str

class GenerateRequest(BaseModel):
    purpose: str = Field(..., description="Purpose of the blog post", min_length=10)
    language: str = Field(..., description="Language for the blog post (e.g., Dutch, English)")
    
    # Determinism controls (optional)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Creativity level (0.0-2.0, default: 0.7)")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")
    max_tokens: Optional[int] = Field(None, ge=100, le=4000, description="Maximum tokens to generate")
    
    @validator('purpose')
    def validate_purpose(cls, v):
        max_length = CONFIG["max_purpose_length"]
        if len(v) > max_length:
            raise ValueError(f"Purpose must be {max_length} characters or less")
        if len(v.strip()) < 10:
            raise ValueError("Purpose must be at least 10 characters")
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        if len(v) > 50:
            raise ValueError("Language name too long")
        return v.strip()

class GenerateResponse(BaseModel):
    markdown: str
    metadata: dict = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    model: str
    documents_loaded: bool
    config: dict

class ConfigResponse(BaseModel):
    example_post_file: str
    company_description_file: str
    documents_dir: str
    model_name: str
    default_tone: str
    default_temperature: float
    max_purpose_length: int

# ============================================
# PDF PROCESSING
# ============================================

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

def find_pdf_files(base_dir: Path) -> tuple:
    """Find PDF files in the documents directory"""
    example_file = CONFIG["example_post_file"]
    company_file = CONFIG["company_description_file"]
    
    example_path = base_dir / example_file
    company_path = base_dir / company_file
    
    if example_path.exists() and company_path.exists():
        return (example_path, company_path)
    
    return (None, None)

@app.on_event("startup")
async def load_documents():
    """Load example post and company description from PDFs"""
    try:
        logger.info("Loading documents with configuration:")
        logger.info(f"  Example post file: {CONFIG['example_post_file']}")
        logger.info(f"  Company description file: {CONFIG['company_description_file']}")
        logger.info(f"  Documents directory: {CONFIG['documents_dir']}")
        logger.info(f"  Default temperature: {CONFIG['default_temperature']}")
        
        base_paths = [
            Path(f"/app/{CONFIG['documents_dir']}"),
            Path(CONFIG['documents_dir']),
            Path(".")
        ]
        
        for base_path in base_paths:
            logger.info(f"  Checking path: {base_path}")
            example_path, company_path = find_pdf_files(base_path)
            
            if example_path and company_path:
                app_state["example_post"] = extract_text_from_pdf(str(example_path))
                app_state["company_description"] = extract_text_from_pdf(str(company_path))
                logger.info(f"  Documents loaded successfully from {base_path}")
                logger.info(f"  Example post: {example_path.name} ({len(app_state['example_post'])} chars)")
                logger.info(f"  Company description: {company_path.name} ({len(app_state['company_description'])} chars)")
                return
        
        error_msg = f"Could not find '{CONFIG['example_post_file']}' and '{CONFIG['company_description_file']}'"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise

# ============================================
# LLM INTERACTION 
# ============================================

async def generate_with_ollama(
    prompt: str, 
    temperature: float = None,
    seed: int = None,
    max_tokens: int = None
) -> tuple[str, dict]:
    """
    Send prompt to Ollama and get response with determinism controls
    Returns: (generated_text, metadata)
    """
    ollama_host = CONFIG["ollama_host"]
    model_name = CONFIG["model_name"]
    
    # Use provided values or fall back to defaults
    if temperature is None:
        temperature = CONFIG["default_temperature"]
    
    # Build options for Ollama
    options = {
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40
    }
    
    # Add seed for determinism (if provided)
    if seed is not None:
        options["seed"] = seed
        logger.info(f"Using seed {seed} for reproducibility")
    
    # Add max tokens (if provided)
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    
    # Log generation parameters
    logger.info(f"Generation params: temperature={temperature}, seed={seed}, max_tokens={max_tokens}")
    
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": options
                }
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            
            # Extract metadata
            metadata = {
                "model": model_name,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
                "generation_time": round(elapsed, 2),
                "total_duration": result.get("total_duration"),
                "load_duration": result.get("load_duration"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count")
            }
            
            return result.get("response", ""), metadata
            
    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {ollama_host}")
        raise HTTPException(status_code=503, detail="LLM service unavailable. Ensure Ollama is running.")
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

def build_prompt(purpose: str, language: str, tone: str) -> str:
    """Construct the prompt for blog generation"""
    
    tone_descriptions = {
        "concise": "brief, to-the-point, no fluff",
        "playful": "friendly, engaging, uses metaphors and light humor",
        "authoritative": "confident, expert, data-driven",
        "professional": "polished, credible, balanced",
        "inspirational": "motivational, encouraging, uplifting"
    }
    
    tone_desc = tone_descriptions.get(tone.lower(), tone)
    
    prompt = f"""You are a professional content writer for the company. Your task is to generate a blog post in Markdown format.

COMPANY CONTEXT:
{app_state['company_description']}

STYLE REFERENCE (example post):
{app_state['example_post']}

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

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    return {
        "example_post_file": CONFIG["example_post_file"],
        "company_description_file": CONFIG["company_description_file"],
        "documents_dir": CONFIG["documents_dir"],
        "model_name": CONFIG["model_name"],
        "default_tone": CONFIG["default_tone"],
        "default_temperature": CONFIG["default_temperature"],
        "max_purpose_length": CONFIG["max_purpose_length"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_host = CONFIG["ollama_host"]
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            ollama_status = response.status_code == 200
    except:
        ollama_status = False
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "model": CONFIG["model_name"],
        "documents_loaded": bool(app_state["example_post"] and app_state["company_description"]),
        "config": {
            "example_post_file": CONFIG["example_post_file"],
            "company_description_file": CONFIG["company_description_file"],
            "documents_dir": CONFIG["documents_dir"],
            "default_temperature": CONFIG["default_temperature"]
        }
    }

@app.post("/settings/tone", response_model=ToneResponse)
async def set_tone(request: ToneRequest):
    """Set the global tone of voice for blog generation"""
    logger.info(f"Setting tone to: {request.tone}")
    app_state["tone"] = request.tone
    return {"tone": request.tone}

@app.post("/generate", response_model=GenerateResponse)
async def generate_blog(request: GenerateRequest):
    """Generate a blog post with determinism controls"""
    
    # Validate documents are loaded
    if not app_state["example_post"] or not app_state["company_description"]:
        raise HTTPException(status_code=500, detail="Documents not loaded")
    
    logger.info(
        f"Generating blog: purpose='{request.purpose[:50]}...', "
        f"language={request.language}, tone={app_state['tone']}, "
        f"temperature={request.temperature}, seed={request.seed}"
    )
    
    start_time = time.time()
    
    # Build prompt
    prompt = build_prompt(
        purpose=request.purpose,
        language=request.language,
        tone=app_state["tone"]
    )
    
    # Generate with determinism controls
    markdown_content, generation_metadata = await generate_with_ollama(
        prompt=prompt,
        temperature=request.temperature,
        seed=request.seed,
        max_tokens=request.max_tokens
    )
    
    elapsed = time.time() - start_time
    
    # Build response metadata
    response_metadata = {
        **generation_metadata,
        "request": {
            "purpose_length": len(request.purpose),
            "language": request.language,
            "tone": app_state["tone"],
            "temperature": request.temperature or CONFIG["default_temperature"],
            "seed": request.seed,
            "max_tokens": request.max_tokens
        },
        "response": {
            "markdown_length": len(markdown_content),
            "word_count": len(markdown_content.split()),
            "total_time": round(elapsed, 2)
        }
    }
    
    logger.info(
        f"Blog generated in {elapsed:.2f}s, "
        f"{len(markdown_content)} chars, "
        f"{response_metadata['response']['word_count']} words"
    )
    
    return {
        "markdown": markdown_content,
        "metadata": response_metadata
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)