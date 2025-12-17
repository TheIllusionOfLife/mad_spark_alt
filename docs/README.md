# Mad Spark Alt Documentation

Welcome to the comprehensive documentation for Mad Spark Alt, a multi-agent idea generation system powered by QADI methodology and dynamic prompt engineering.

## 📚 Documentation Overview

This documentation is organized into focused guides for different audiences and use cases. For project overview and installation, start with the [main README](../README.md).

## 🗂️ Documentation Structure

### 📖 User Guides
- **[CLI Usage Guide](cli_usage.md)** - Complete command-line interface reference
  - QADI multi-agent analysis with auto question type detection
  - CLI flags: `--type`, `--concrete`, evolution commands
  - Environment setup and API key configuration
  
- **[Examples Guide](examples.md)** - Comprehensive code samples and usage patterns
  - Dynamic prompt engineering examples
  - Question type detection and adaptive prompts
  - Python API usage patterns
  - Real-world use cases

### 🔧 Developer Reference
- **[QADI API Documentation](qadi_api.md)** - Complete technical API reference
  - Core interfaces and data models
  - Dynamic prompt engineering modules
  - QADI orchestration and agent systems
  - Evolution and evaluation APIs

### 🏗️ Development Resources
- **[CLAUDE.md](../CLAUDE.md)** - AI development assistant instructions
- **[DEVELOPMENT.md](../DEVELOPMENT.md)** - Development setup and guidelines
- **[Tests Directory](../tests/)** - Test suite and examples

## 🎯 Quick Navigation by Role

### 👤 New Users
1. Start with [Project README](../README.md) for installation and overview
2. Try the [CLI Usage Guide](cli_usage.md) for command examples
3. Explore [Examples Guide](examples.md) for code samples

### 👨‍💻 Developers
1. Review [API Documentation](qadi_api.md) for technical details
2. Check [DEVELOPMENT.md](../DEVELOPMENT.md) for setup and patterns
3. Study [Examples Guide](examples.md) for implementation patterns

### 🔬 Researchers
1. Read [RESEARCH.md](../RESEARCH.md) for academic background
2. Explore [API Documentation](qadi_api.md) for system architecture
3. Review test files in [../tests/](../tests/) for validation approaches

## 🚀 Quick Start Links

- **Try QADI**: `msa "Your question"`
- **View all CLI options**: `msa --help`
- **Run evolution**: `msa "Your problem" --evolve`
- **See examples**: [examples.md](examples.md)

## 🧭 Feature-Specific Documentation

### QADI System
- **Technical details**: [qadi_api.md](qadi_api.md)
- **Usage examples**: [examples.md](examples.md)
- **CLI usage**: [cli_usage.md](cli_usage.md)

### Genetic Evolution
- **API reference**: [qadi_api.md](qadi_api.md) (evolution section)
- **CLI commands**: [cli_usage.md#genetic-evolution-cli](cli_usage.md#genetic-evolution-cli)
- **Examples**: [examples.md](examples.md) and `examples/evolution_demo.py`

### Multi-Agent QADI
- **Core concepts**: [Project README](../README.md#dynamic-prompt-engineering)
- **API details**: [qadi_api.md#qadi-orchestration](qadi_api.md#qadi-orchestration)
- **Usage patterns**: [examples.md](examples.md)

## 📂 Related Files

### Documentation Files
```
docs/
├── README.md           # This file - documentation index
├── cli_usage.md        # Complete CLI reference
├── examples.md         # Code samples and usage patterns
└── qadi_api.md         # Technical API documentation
```

### Project Files
```
../
├── README.md           # Main project overview and installation
├── CLAUDE.md           # AI development instructions
├── DEVELOPMENT.md      # Development setup and guidelines
├── RESEARCH.md         # Academic background and methodology
└── examples/           # Runnable example scripts
```

## 🔄 Documentation Updates

This documentation is actively maintained and reflects the current system capabilities:

- **✅ Dynamic Prompt Engineering**: 100% auto-detection accuracy
- **✅ LLM Integration**: Google Gemini API
- **✅ Genetic Evolution**: Complete with parallel evaluation
- **✅ CLI Interface**: Full command-line tools with rich output

Last updated: December 2025

---

*For bug reports, feature requests, or documentation improvements, please use the project's GitHub issues.*