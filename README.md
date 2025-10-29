# Blog Generator API

This project provides a robust, containerized FastAPI application for generating contextualized blog posts in Markdown format using an open-source Large Language Model (LLM) served by Ollama.

The application incorporates context from external PDF documents and a user-defined tone of voice to produce highly relevant content.

## 🚀 How to Build and Run the Container

The entire application stack—FastAPI, Ollama, and the LLM—is orchestrated using Docker Compose.

**Prerequisites:** Docker Engine (or Docker Desktop) installed and running.

1. **Clone/Setup:** Ensure you have the `docker-compose.yml`, `Dockerfile`, the Python source code (`main.py`), and the context PDF files(documents folder) in the same directory.

2. **Build and Run:** Execute the following command to build the API image and start both services in the background.

   ```
   docker compose up -d --build
   ```

3. **Monitor Startup:** The API container will automatically pull the required LLM model (`llama3.2:3b`) on first run. This process can take several minutes. Use the logs to monitor progress:

   ```
   docker compose logs -f
   ```

4. **Verify Health:** Once the startup logs show Uvicorn is running, confirm the services are healthy.

   ```
   curl http://127.0.0.1:8080/health
   ```

   **Expected Response:**

   ```
   {"status":"healthy","model":"llama3.2:3b","documents_loaded":true,"config":{"example_post_file":"example_post.pdf","company_description_file":"company_description.pdf","documents_dir":"documents","default_temperature":0.7}}
   ```

5. **Get current configurations:**

   ```
   curl http://127.0.0.1:8080/config
   ```

   **Expected Response:**

   ```
   {"example_post_file":"example_post.pdf","company_description_file":"company_description.pdf","documents_dir":"documents","model_name":"llama3.2:3b","default_tone":"professional","default_temperature":0.7,"max_purpose_length":500}
   ```

You can access the interactive docs in your browser at `http://127.0.0.1:8080/docs`.

## Model Rationale

|                     |                           |
| ------------------- | ------------------------- |
| **Parameter**       | **Value**                 |
| **Name/Version**    | `llama3.2:3b`             |
| **Parameter Count** | Approx. 3 Billion         |
| **License**         | Llama 3 Community License |

### Model Choice

The **Llama 3.2 3B** parameter model was selected as an optimal choice for a containerized API environment. It provides a superior balance of **quality** for nuanced tasks (like integrating complex context, adhering to tone, and generating coherent Markdown) and **performance/resource efficiency** (low latency for API calls) compared to larger, more resource-intensive models, making it ideal for rapid deployment and testing.

## 📝 Example Usage

The API exposes two primary endpoints on `http://127.0.0.1:8080`.

### 1. Set Global Tone of Voice

Sets the content generation style. Available tones include: `authoritative`, `concise`, `playful`, `inspirational` and `professional`.

```
curl -X POST http://localhost:8080/settings/tone -H "Content-Type: application/json" -d "{\"tone\": \"authoritative\"}"

```

**Expected Response (HTTP 200):**

```
{"tone": "authoritative"}
```

### 2. Generate Blog Post

Generates the blog post using the current global tone, the purpose, the requested language, and the content drawn from the two contextual PDFs (`example_post.pdf` and `company_description.pdf`).

```
curl -X POST http://localhost:8080/generate  -H 'Content-Type: application/json'  -d '{ "purpose": "Introduce our new analytics feature to SMB marketers", "language": "English" }'

```

**Expected Response (HTTP 200):**

```
{
  "markdown": "# Introducing InsightPulse: ..."
}
```

### 3. Default Behavior

```
# Uses default temperature (0.7), no seed
curl -X POST http://localhost:8080/generate  -H 'Content-Type: application/json'  -d '{ "purpose": "Explain why home batteries are smart", "language": "English" }'
```

**Expected Response (HTTP 200):**

```
{
  { "markdown": "# Why Home Batteries Are a Smart Investment\n\n## Introduction\n\nHome batteries are becoming increasingly popular...", "metadata": { "model": "llama3.2:3b", "temperature": 0.7, "seed": null, "generation_time": 45, "word_count": 234 } }
}
```

### 4. Set temperature and seed

```
curl -X POST http://localhost:8080/generate  -H 'Content-Type: application/json'  -d '{ "purpose": "Quick solar panel tips", "language": "English", "temperature": 0.0, "seed": 42 }'
```

**Expected Response:**

```
{
  "markdown":"# Quick Solar Panel Tips to Save You Money and Energy\n\nIntroduction\n------------\n\nAs more and more homeowners invest in solar panels, it's essential to consider how to maximize their energy savings. One often-overlooked component of a solar panel system is the home battery..."
}
```

### 5. Set a language

```
 curl -X POST http://localhost:8080/generate  -H 'Content-Type: application/json'  -d '{ "purpose": "Explain home batteries", "language": "Dutch", "temperature": 0.5, "seed": 42, "max_tokens": 1000 }'
```

**Expected Response:**

```
{
  "markdown":"# Home Batterijen: Hoe werken ze en waarom zijn ze zo populair?\n\n## Inleiding\n\nHome batterijen zijn momenteel steeds meer populair onder mensen die zonnenpaneelen hebben. Maar hoe werken deze batterijen exact? Laat ons kijk naar hoe home batterijen interactie hebben met zonnenpaneelen en wat de voordelen zijn...."
}
```
