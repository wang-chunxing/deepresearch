# Deep Research Report Generation Agent System

An advanced AI system for generating comprehensive research reports using multi-agent collaboration and long-term memory management, based on the architecture design from the research report.

## Architecture Overview

This system implements a five-layer architecture:

1. **Perception Layer**: Handles user input and initial understanding
2. **Memory Layer**: Implements MemGPT's layered memory architecture with main context and external storage
3. **Reasoning Layer**: Multi-agent collaboration with specialized agents (planning, research, analysis, validation)
4. **Generation Layer**: Report template engine with structured output
5. **Interaction Layer**: API endpoints with event flow recording

## Features

- Long-term context management using layered memory architecture
- Multi-source information integration with web search and scraping
- Multi-agent collaboration for enhanced reasoning
- Structured report generation in multiple formats (Markdown, HTML)
- Source validation and credibility assessment
- Comprehensive logging and monitoring
- Asynchronous processing with task management

## Project Structure

```
/workspace/
├── config.py                 # Configuration settings
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── design_docs/              # Design documentation
│   ├── requirements_specification.md
│   ├── system_architecture_design.md
│   ├── implementation_roadmap.md
│   └── summary_and_consistency_check.md
└── src/                     # Source code
    ├── __init__.py
    ├── api/                  # API layer
    │   ├── __init__.py
    │   └── api_router.py
    ├── agents/               # Reasoning layer
    │   ├── __init__.py
    │   └── research_agent.py
    ├── memory/               # Memory layer
    │   ├── __init__.py
    │   └── memory_manager.py
    ├── generation/           # Generation layer
    │   ├── __init__.py
    │   └── report_generator.py
    └── tools/                # Tool layer
        ├── __init__.py
        └── web_search_tool.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deep-research-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Usage

1. Start the API server:
   ```bash
   python main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Use the API endpoints:
   - `POST /api/v1/research` - Start a new research task
   - `GET /api/v1/research/{task_id}` - Get research result
   - `GET /api/v1/research/{task_id}/status` - Get task status
   - `GET /api/v1/research/history` - Get research history
   - `GET /health` - Health check

## API Example

```bash
# Start a research task
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Impact of climate change on polar ice caps",
    "max_sources": 10,
    "report_format": "markdown",
    "include_sources": true,
    "depth": "comprehensive"
  }'
```

## Configuration

The system can be configured via environment variables in the `.env` file:

- `LLM_BASE_MODEL`: LLM model to use (default: gpt-4-turbo)
- `LLM_TEMPERATURE`: Temperature for LLM (default: 0.7)
- `VECTOR_DB_TYPE`: Vector database type (default: chroma)
- `API_HOST`: Host for the API server (default: 0.0.0.0)
- `API_PORT`: Port for the API server (default: 8000)
- `DEBUG_MODE`: Enable debug mode (default: False)

## Implementation Roadmap

Based on the implementation roadmap document, this system represents Phase 1 (Basic Architecture) of the four-phase development plan:

1. **Phase 1: Basic Architecture** (Completed) - Core components and basic functionality
2. **Phase 2: Reasoning Enhancement** - Multi-agent debate and dynamic planning
3. **Phase 3: Report Optimization** - Scientific domain processing and citation management
4. **Phase 4: System Integration** - Performance optimization and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
