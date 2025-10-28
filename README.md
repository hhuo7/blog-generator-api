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

   **Expected Response:** `{"status":"healthy","model":"llama3.2:3b","documents_loaded":true}`

## Model Rationale

|                     |                           |
| ------------------- | ------------------------- |
| **Parameter**       | **Value**                 |
| **Name/Version**    | `llama3.2:3b`             |
| **Parameter Count** | Approx. 3 Billion         |
| **License**         | Llama 3 Community License |

### Model Choice

The **Llama 3.2 3B** parameter model was selected as an optimal choice for a containerized API environment. It provides a superior balance of **quality** for nuanced tasks (like integrating complex context, adhering to tone, and generating coherent Markdown) and **performance/resource efficiency** (low latency for API calls) compared to larger, more resource-intensive models, making it ideal for rapid deployment and testing.

## 📝 Example $\text{curl}$ Requests

The API exposes two primary endpoints on `http://127.0.0.1:8080`.

### 1. Set Global Tone of Voice

Sets the content generation style. Available tones include: `authoritative`, `concise`, `playful`, and `professional`.

```
curl -X POST http://127.0.0.1:8080/settings/tone \
  -H 'Content-Type: application/json' \
  -d '{"tone": "authoritative"}'
```

**Expected Response (HTTP 200):**

```
{"tone": "authoritative"}
```

### 2. Generate Blog Post

Generates the blog post using the current global tone, the purpose, the requested language, and the content drawn from the two contextual PDFs (`example_post.pdf` and `company_description.pdf`).

```
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"purpose":"Introduce our new analytics feature to SMB marketers","language":"English"}'
```

**Expected Response (HTTP 200):**

```
{
  "markdown": "# Introducing InsightPulse: ..."
}
```
