## Design Philosophy

Astral's design philosophy revolves around simplicity, consistency, flexibility, and performance, empowering developers to build rapidly, confidently, and innovatively. We achieve this through clearly defined core principles:

### 1. Fully Abstracted, Type-Safe Interfaces

Astral provides type-safe, fully abstracted interfaces for interacting with core AI capabilities including completions, embeddings, image and video generation, real-time audio, speech-to-text, and text-to-speech. This abstraction simplifies integration, removing the need for developers to handle provider-specific nuances and enabling effortless transitions between multiple providers.

### 2. Provider Encapsulation

Each AI provider's specific implementations are encapsulated within dedicated modules (e.g., `OpenAI`, `Anthropic`, and `Gemini`). These modules inherit from a base adapter class, isolating provider-specific details and enabling efficient translation between Astral’s generic interface and each provider’s unique implementation. Additionally, an authentication metaclass ensures users can flexibly utilize any authentication method supported by each provider without introducing tight coupling between authentication and core functionalities. Authentication details can be configured either through code or via an `auth_config.yaml` file. A base client provider protocol further standardizes interactions with provider APIs.

### 3. Intelligent Resource Standardization

Astral standardizes critical AI resources across all supported providers, eliminating provider-specific inconsistencies. By maintaining a unified resource interface, developers can effortlessly interact with and utilize resources across providers:

- Completions
- Embeddings
- Image & Video Generation
- Real-time Audio
- Speech-to-Text / Text-to-Speech

This standardization accelerates development by allowing developers to focus solely on application logic rather than provider intricacies. We provide both a top-level class and function for interacting with these resources. For example, to call a completions endpoint, developers can use either the `completion` function or instantiate the `Completions` class and call `run` or `arun` (to execute asynchronously).

### 4. Clear Messaging and Advanced Prompt Management

Astral delivers a clear and unified messaging interface for completion models to simplify communication across providers. We are actively developing prompt management capabilities, including templating, versioning, and comprehensive prompt lifecycle management, to meet enterprise-level requirements.

### 5. Automatically Updated, IDE-Friendly Type Definitions

Astral leverages automatically generated files to guarantee real-time accuracy of model availability and parameters, immediately reflecting provider updates. This ensures developers benefit from precise, context-aware type hinting directly within their IDEs.

### 6. Managed and Flexible Client Connections

Client management is fully handled within Astral for common use cases, optimizing performance by establishing and reusing TCP connections across multiple requests. At the same time, Astral offers complete flexibility, allowing developers to manage connections directly when specific scenarios or advanced customizations are required.

### 7. Lightweight and Performant by Design

Astral is engineered for simplicity and performance, ensuring seamless integration into your existing tech stack without unnecessary complexity. Designed specifically to prioritize speed and scalability, Astral enhances your development workflow rather than slowing it down. Easily customize your setup by adding only the providers you need using the Astral CLI or your preferred Python package manager. This modular approach guarantees minimal overhead, ensuring your broader application remains optimized and responsive.

## Contributing to Astral

We welcome contributions from developers, designers, and AI enthusiasts passionate about shaping the future of AI applications and workflows. Whether you're looking to fix a bug, implement a new feature, or enhance documentation, your efforts directly strengthen and empower the Astral community.

### 🌌 Join the Astral Community!

Before diving in, make sure to:

1. **⭐️ Star the Repository!**
    
    Loving Astral? Show your support by starring the [repo](https://github.com/chrismaresca/Astral-AI)—it helps new folks discover us and expands our stellar community!
    
2. **💬 Join our Discord Community:**
    
    Hop into our vibrant [Discord community](https://discord.gg/czNPugfa) to connect with fellow contributors, brainstorm ideas, ask questions, or just hang out! It's the best place to stay updated on everything Astral. 
    
    Thanks for helping shape the future of Astral—we can't wait to see what you build!
    

### **🚀 Getting Started Contributing**

Ready to build something awesome? Follow these steps:

1. **Fork and Clone:**
    
    Kick things off by forking the repo to your GitHub account and cloning your fork locally:
    
    ```bash
    # Fork the repo first from GitHub, then clone your fork
    git clone https://github.com/YOUR-GITHUB-USERNAME/Astral-AI.git
    cd astral-ai
    ```
    
2. **Set Up Development Environment:**
    
    We use [**UV**](https://docs.astral.sh/uv/)—a blazing-fast Python package and project manager built with Rust—to streamline setup.
    
    ```bash
    uv sync
    ```
    
3. **Create a New Branch:**
    
    Create a new branch named clearly after your contribution:
    
    ```
    git checkout -b feature/my-feature-name
    ```
    

### **📌 Contribution Guidelines**

- **Feature Contributions:**
    
    Before beginning work, open an issue to clearly outline your idea, its motivation, intended use cases, and your planned implementation approach.
    
- **Bug Reports and Fixes:**
    
    When reporting bugs, please include reproducible examples, relevant error logs, and environment details. Reference corresponding issues clearly in your pull requests.
    
- **Documentation Improvements:**
    
    Help make Astral easier to use! Improve clarity, add examples, or refine explanations to make life easier for future contributors.
    

### **📜** Commit and Pull Request (PR) Guidelines

- **Commit Messages:**
    
    Use clear, concise, imperative-form commit messages (e.g., "Add completion interface for Gemini").
    
- **Pull Requests:**
    
    Submit focused PRs addressing a single feature or bug. Clearly summarize your changes, reference relevant issues, and specify areas needing particular attention.
    

### **🧪** Testing Guidelines

Astral emphasizes comprehensive, verbose testing to ensure clarity, robustness, and maintainability. Our testing strategy includes:

- **Verbose Unit Tests:** Clearly document the intent, scenarios, and expected behavior using descriptive test names and assertions. Each test should clearly communicate its purpose and expected outcome. *We are huge fans of building with AI. Use AI to build tests.*
- **Integration Tests:**
    
    Integration tests verify that Astral correctly interacts with provider APIs and resources in real-world scenarios. Place these tests within the `tests/integration/` directory, structured by resources and providers. Integration tests should:
    
    - Clearly indicate the scenario tested, provider interactions, and expected results.
    - Include thorough validation for response structure, error handling, and edge cases.
    - Ensure you use mocks or provider sandboxes when possible to avoid consuming API quotas.
- **Structured Test Directory:**
    
    The testing folder structure should directly reflect the main project's hierarchy:
    

```
tests/
├── _types/
│   ├── test_request.py
│   └── test_response.py
├── integration/
│   ├── providers/
│   └── resources/
├── completions/
│   └── test_openai_completions.py
├── embeddings/
│   └── test_anthropic_embeddings.py
└── utils/
    └── test_cost_utils.py
```

Ensure your tests pass before submitting your PR:

```
pytest tests/
```

### **🔍 Review Process**

After submitting your PR, our core team will provide detailed feedback. Engage actively with reviewers and promptly address comments or suggestions. Upon approval, your changes will be merged.

### 🚀 Let’s Build the Future of AI—Together!

Thanks for being part of the Astral community! We look forward to seeing your contributions. 