<p align="center">
  <img src="docs/images/embedding_studio_logo.svg" alt="Embedding Studio" />
</p>

<p align="center">
<a href="https://hugsearch.demo.embeddingstud.io/" style="font-size: 20px;"><strong>👉 Try the Live Demo</strong></a>
</p>

<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="version"></a>
    <a href="https://www.python.org/downloads/release/python-31014/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
    <a href="#"><img src="https://img.shields.io/badge/CUDA-11.7.1-green.svg" alt="CUDA 11.7.1"></a>
    <a href="#"><img src="https://img.shields.io/badge/docker--compose-2.17.0-blue.svg" alt="Docker Compose Version"></a>
</p>

<p align="center">
    <a href="https://embeddingstud.io/">Website</a> •
    <a href="https://embeddingstud.io/tutorial/getting_started/">Documentation</a> •
    <a href="https://embeddingstud.io/challenges/">Challenges & Solutions</a> •
    <a href="https://embeddingstud.io/challenges/">Use Cases</a>
</p>

**Embedding Studio** is an innovative open-source framework designed to transform embedding models and vector databases into comprehensive, self-improving search engines. With built-in clickstream collection, continuous model refinement, and intelligent vector optimization, it creates a feedback loop that enhances search quality over time based on real user interactions.

<table style="margin-left: auto; margin-right: auto">
    <tr align="center"><td>Community Support</td></tr>
    <tr align="center">
        <td>
            Embedding Studio grows with our team's enthusiasm. Your <b>star on the repository</b> helps us keep developing. <br>
            Join us in reaching our goal:
            <p align="center">
                <a href="#"><img src="https://embeddingstud.io/badge?title=Stars%20Goal&scale=500&width=200&color=5d5d5d" alt="Progress"/></a>
            </p>
        </td>
    </tr>
</table>

## Features

### Core Capabilities

1. 🔄 **Full-Cycle Search Engine** - Transform your vector database into a complete search solution
2. 🖱️ **User Feedback Collection** - Automatically gather clickstream and session data
3. 🚀 **Continuous Improvement** - Enhance search quality on-the-fly without long waiting periods
4. 📊 **Performance Monitoring** - Track search quality metrics through comprehensive dashboards
5. 🎯 **Iterative Fine-Tuning** - Improve your embedding model through user interaction data
6. 🔍 **Blue-Green Deployment** - Zero-downtime deployment of improved embedding models
7. 💾 **Multi-Source Integration** - Connect to various data sources (S3, GCP, PostgreSQL, etc.)
8. 🧠 **Vector Optimization** - Apply post-training adjustments for incremental improvements

### Specialized Features

- 📈 **Personalization Support** - Create user-specific vector adjustments based on individual behavior
- 💬 **Suggestion System** - Generate intelligent query autocompletions based on user patterns
- 🔎 **Category Prediction** - Automatically identify relevant categories for search queries
- 🔤 **Multi-Modal Support** - Work with text, images, and structured data in one framework
- 🧩 **Plugin Architecture** - Extend functionality through a comprehensive plugin system

### In Development (*)

- 📑 **Zero-Shot Query Parser** - Mix structured and unstructured search queries
- 📚 **Catalog Pre-Training** - Fine-tune embedding models on your specific content before deployment
- 📊 **Advanced Analytics** - More detailed insights into search performance and user behavior

(*) - Features in active development

## When is Embedding Studio the Best Fit?

More about it [here](docs/when-to-use-the-embeddingstudio.md).

- 📚💼 **Rich Content Collections** - Businesses with extensive catalogs and unstructured data
- 🛍️🤝 **Customer-Centric Platforms** - Applications prioritizing personalized user experiences
- 🔄📊 **Dynamic Content** - Platforms with evolving content and changing user preferences
- 🔍🧠 **Complex Queries** - Systems handling nuanced and multifaceted search needs
- 🔄📊 **Mixed Data Types** - Applications integrating different data formats in search
- 🔄🚀 **Continuous Improvement** - Platforms seeking ongoing optimization through user interactions
- 💵💡 **Cost-Conscious Organizations** - Teams looking for powerful yet affordable solutions

## Challenges Solved

**Disclaimer:** Embedding Studio is not another Vector Database - it's a framework that transforms your Vector Database into a complete Search Engine with all necessary components.

- ✅ **Cold Start Problems** - Jump-start search quality with minimal data
- ✅ **Static Search Quality** - Create systems that improve automatically over time
- ✅ **Long Improvement Cycles** - Reduce frustration with fast feedback loops
- ✅ **Resource-Heavy Reindexing** - Optimize the updating process for better performance
- ✅ **Hybrid Search Complexity** - Seamlessly combine structured and unstructured search
- ✅ **Query Understanding** - Parse natural language queries more effectively
- ✅ **New Content Discovery** - Ensure fresh items get proper visibility

More about challenges and solutions [here](https://embeddingstud.io/challenges/)

## System Architecture

Embedding Studio uses a modular, service-based architecture:

### Core Components

- **API Service** - Central coordination point for applications
- **Vector Database** - PostgreSQL with pgvector for embedding storage
- **Clickstream System** - Captures and processes user interactions
- **Worker Services**:
  - **Fine-Tuning Worker** - Handles model training and improvement
  - **Inference Worker** - Manages Triton Inference Server for embeddings
  - **Improvement Worker** - Processes incremental vector adjustments
  - **Upsertion Worker** - Manages content updates and indexing

### Data Flow

1. **Content Ingestion** - Load data from various sources
2. **User Interaction** - Collect clickstream data through API endpoints
3. **Fine-Tuning** - Use interaction data to improve embedding models
4. **Model Deployment** - Update inference service with improved models
5. **Search and Retrieval** - Deliver better results based on fine-tuned models

## Comparison with Traditional Approaches

![Embedding Studio Chart](assets/embedding_studio_chart.png)

Our framework enables you to continuously fine-tune your model based on user experience, allowing you to form search 
results for user queries faster and more accurately.

$${\color{red}RED:}$$ On the graph, typical search solutions without enhancements, 
such as Full Text Searching (FTS), Nearest Neighbor Search (NNS), and others, are marked in red. Without the use of 
additional tools, the search quality remains unchanged over time.

$${\color{orange}ORANGE:}$$ Solutions are depicted that accumulate some feedback (clicks, reviews, votes, discussions, etc.) and then
initiate a full model retraining. The primary issue with these solutions is that full model retraining is a
time-consuming and expensive procedure, thus lacking reactive adjustments (for example, when a product suddenly
experiences increased demand, and the search system has not yet adapted to it).

$${\color{#6666ff}INDIGO:}$$ We propose a solution that allows collecting user feedback and rapidly retraining the model on the difference between
the old and new versions. This enables a smoother and more relevant search quality curve for your system.

## Getting Started

### Prerequisites

- Docker Compose v2.17.0+
- For fine-tuning: NVIDIA GPU with CUDA support
- Minimum 8GB RAM allocated to Docker

## Documentation

For comprehensive documentation:

- [Core Concepts](docs/tutorial/getting_started/core_concepts.md)
- [Architecture Overview](docs/tutorial/getting_started/architecture_overview.md)
- [Docker Quick Start](docs/tutorial/getting_started/docker_quickstart.md)
- [Configuration Guide](docs/tutorial/getting_started/configurations.md)
- [Plugin Development](docs/tutorial/plugins/understanding_plugin_system.md)
- [Vector Database Integration](docs/tutorial/vectordb/integration.md)
- [Code Documentation](docs/embedding_studio)

## Plugin System

Embedding Studio features a powerful plugin architecture allowing extension of:

- Data loaders for different sources
- Text and image processors
- Fine-tuning methods
- Vector optimization strategies
- Query processing logic

Create custom plugins by extending base classes and implementing your specific logic.

## Contributing

We welcome contributions to Embedding Studio! To contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please check our contributing guidelines for detailed information.

## 📬 Contact Us
<strong>EulerSearch Inc.</strong><br/> 3416, 1007 N Orange St. 4th Floor,<br/> Wilmington, DE, New Castle, US, 19801<br/> Contact Email: <a href="mailto:aleksandr.iudaev@eulersearch.com">aleksandr.iudaev@eulersearch.com</a><br/> Phone: +34 (691) 454 148<br/> LinkedIn: <a href="https://www.linkedin.com/in/alexanderyudaev/">https://www.linkedin.com/in/alexanderyudaev/</a>

## License

Embedding Studio is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.