# Multimodal Features: URL Context, Image Understanding, Document Processing

**Date:** 2025-11-08
**Status:** Planning Phase
**Target:** Add multimodal capabilities to Mad Spark Alt
**Estimated Timeline:** 10-15 days across 5 phases

---

## 🎯 Executive Summary

### Vision

Transform Mad Spark Alt from a text-only QADI system to a **multimodal reasoning platform** that can analyze images, documents, and web content. This leverages Gemini's native multimodal capabilities to unlock new use cases:

- **Visual Analysis**: Architecture diagrams, product screenshots, data visualizations
- **Document Processing**: Research papers, reports, specifications (PDF, up to 1000 pages)
- **Web Context**: Fetch and analyze content from URLs (up to 20 URLs, 34MB each)
- **Hybrid Reasoning**: Combine text, images, documents, and URLs in single QADI cycle

### Strategic Trade-off

**Gemini Lock-in**: Implementing these features creates **strong coupling to Google's Gemini API**. While OpenAI, Anthropic, and other providers offer similar capabilities, their APIs differ significantly:

| Feature                  | Gemini                            | OpenAI GPT-4V              | Anthropic Claude 3.5     |
| ------------------------ | --------------------------------- | -------------------------- | ------------------------ |
| **URL Context**          | Native tool (`url_context`)       | Manual fetch required      | Manual fetch required    |
| **Image Input**          | Inline data + File API            | Inline base64 + URLs       | Inline base64 only       |
| **Document Processing**  | Native PDF vision (1000 pages)    | Vision API (limited pages) | PDF via conversion       |
| **Request Structure**    | `contents` array with parts       | `messages` with `image_url`| `messages` with base64   |
| **File API**             | Yes (50MB, 48hr retention)        | No (inline only)           | No (inline only)         |

**Decision**: Accept Gemini lock-in for v1.0, design abstraction layer for future multi-provider support.

---

## 📚 Feature Specifications

### 1. URL Context

**Capability**: Fetch and analyze content from up to 20 URLs per request.

**API Details** (from [docs](https://ai.google.dev/gemini-api/docs/url-context)):
- **Tool activation**: `tools = [{"url_context": {}}]`
- **URL inclusion**: Include URLs directly in prompt text
- **Supported content**: HTML, JSON, XML, PDF, PNG, JPEG, WebP (up to 34MB each)
- **Response metadata**: `url_context_metadata` shows retrieval status per URL

**Limitations**:
- ❌ No paywalled content
- ❌ No YouTube videos
- ❌ No Google Workspace files
- ❌ No audio/video files
- ⚠️ Content counts toward input token limit

**Use Cases**:
```bash
# Research synthesis
uv run mad_spark_alt "Generate hypotheses about practical applications" \
  --url https://arxiv.org/pdf/2303.12712.pdf

# Competitive analysis
uv run mad_spark_alt "Compare feature sets and pricing" \
  --url https://competitor-a.com/pricing \
  --url https://competitor-b.com/pricing

# Documentation-driven development
uv run mad_spark_alt "How can we implement this API?" \
  --url https://api.stripe.com/docs/payments
```

---

### 2. Image Understanding

**Capability**: Analyze images with vision + reasoning capabilities.

**API Details** (from [docs](https://ai.google.dev/gemini-api/docs/image-understanding)):
- **Formats**: PNG, JPEG, WebP, HEIC, HEIF
- **Sources**: Local files, URLs, File API uploads
- **Size limits**: 20MB inline, no limit via File API
- **Token cost**: ≤384px = 258 tokens, larger images tiled at 258 tokens/768px chunk
- **Max images**: 3,600 per request (Gemini 2.5/2.0/1.5)

**Best Practices**:
- ✅ Correct rotation before upload
- ✅ Clear, non-blurry images
- ✅ Place text prompt *after* image in single-image requests
- ✅ Use File API for repeated reuse or >20MB files

**Advanced Features** (Gemini 2.0+):
- **Object detection**: Bounding boxes with normalized coordinates (0-1000)
- **Segmentation**: Contour masks as base64 PNG (Gemini 2.5+)

**Use Cases**:
```bash
# Architecture review
uv run mad_spark_alt "Suggest improvements to this system design" \
  --image architecture_diagram.png

# Product comparison
uv run mad_spark_alt "What differentiates these products?" \
  --image product_a.jpg --image product_b.jpg

# Data visualization analysis
uv run mad_spark_alt "Identify trends and anomalies" \
  --image sales_chart.png

# Real-world problem solving
uv run mad_spark_alt "How can we improve traffic flow at this intersection?" \
  --image intersection_photo.jpg
```

---

### 3. Document Processing

**Capability**: Process PDFs with native vision understanding of text, images, diagrams, charts, and tables.

**API Details** (from [docs](https://ai.google.dev/gemini-api/docs/document-processing)):
- **Primary format**: PDF with full visual understanding
- **Other formats**: TXT, Markdown, HTML, XML (text extraction only, no rendering)
- **Page limit**: 1,000 pages per request
- **Size limits**: 20MB inline, 50MB via File API
- **Token cost**: ~258 tokens per page
- **File API retention**: 48 hours

**Best Practices**:
- ✅ Correct page rotation before upload
- ✅ Avoid blurry pages
- ✅ Place text prompt after document for single-doc requests
- ✅ Use File API for documents >20MB

**Special Capabilities**:
- Process multiple PDFs (up to 1000 total pages) in single request
- Generate structured output from document content
- Transcribe to alternative formats preserving layout
- Answer questions integrating visual and textual elements

**Use Cases**:
```bash
# Research paper analysis
uv run mad_spark_alt "Summarize methodology and key findings" \
  --document research_paper.pdf

# Technical specification review
uv run mad_spark_alt "Identify potential implementation challenges" \
  --document api_spec.pdf

# Multi-document synthesis
uv run mad_spark_alt "Compare approaches across these papers" \
  --document paper1.pdf --document paper2.pdf --document paper3.pdf

# Contract analysis
uv run mad_spark_alt "Extract key terms and obligations" \
  --document contract.pdf
```

---

## 🏗️ Architecture Design

### Provider Abstraction Strategy

**Challenge**: Gemini's multimodal API differs from OpenAI/Anthropic.

**Solution**: Abstract multimodal inputs at the `LLMRequest` level, provider-specific translation at implementation.

```
┌─────────────────────────────────────────────────┐
│ CLI / Orchestrator Layer                       │
│ (Provider-agnostic multimodal inputs)          │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ LLMRequest (Abstract Interface)                │
│ - user_prompt: str                              │
│ - system_prompt: Optional[str]                  │
│ - multimodal_inputs: Optional[List[MultimodalInput]] │
│ - urls: Optional[List[str]]                     │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ GeminiProvider   │  │ Future: OpenAI   │
│ - contents array │  │ - messages array │
│ - parts system   │  │ - image_url      │
│ - url_context    │  │ - manual fetch   │
│ - File API       │  │ - base64 only    │
└──────────────────┘  └──────────────────┘
```

### Data Structures

**Core Abstraction**:
```python
# src/mad_spark_alt/core/multimodal.py

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

class MultimodalSourceType(Enum):
    """Source types for multimodal inputs"""
    FILE_PATH = "file_path"        # Local file system
    URL = "url"                    # Remote URL
    BASE64 = "base64"              # Inline base64-encoded
    FILE_API = "file_api"          # Pre-uploaded to provider's File API

class MultimodalInputType(Enum):
    """Types of multimodal inputs"""
    IMAGE = "image"                # PNG, JPEG, WebP, etc.
    DOCUMENT = "document"          # PDF, DOCX, etc.
    AUDIO = "audio"                # For future audio support
    VIDEO = "video"                # For future video support

@dataclass
class MultimodalInput:
    """
    Unified multimodal input representation.

    Provider-agnostic structure that gets translated to
    provider-specific format by LLMProvider implementations.
    """
    input_type: MultimodalInputType
    source_type: MultimodalSourceType
    data: str  # File path, URL, base64 string, or File API ID
    mime_type: str  # e.g., "image/jpeg", "application/pdf"

    # Optional metadata
    description: Optional[str] = None  # User-provided description
    file_size: Optional[int] = None    # Size in bytes (for validation)
    page_count: Optional[int] = None   # For documents

    def validate(self) -> None:
        """Validate input constraints"""
        if self.input_type == MultimodalInputType.IMAGE:
            self._validate_image()
        elif self.input_type == MultimodalInputType.DOCUMENT:
            self._validate_document()

    def _validate_image(self) -> None:
        """Validate image-specific constraints"""
        if self.source_type == MultimodalSourceType.BASE64:
            # Gemini: 20MB limit for inline
            if self.file_size and self.file_size > 20 * 1024 * 1024:
                raise ValueError(f"Image too large for inline: {self.file_size} bytes (max 20MB)")

        # Validate MIME type
        valid_image_types = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
        if self.mime_type not in valid_image_types:
            raise ValueError(f"Unsupported image type: {self.mime_type}")

    def _validate_document(self) -> None:
        """Validate document-specific constraints"""
        if self.page_count and self.page_count > 1000:
            raise ValueError(f"Document too long: {self.page_count} pages (max 1000)")

        # Gemini primarily supports PDF for vision understanding
        if self.mime_type not in ["application/pdf", "text/plain", "text/markdown", "text/html"]:
            raise ValueError(f"Unsupported document type: {self.mime_type}")
```

**Extended LLMRequest**:
```python
# src/mad_spark_alt/core/llm_provider.py

@dataclass
class LLMRequest:
    """Request for LLM generation with multimodal support"""

    # Text inputs (existing)
    user_prompt: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None

    # Structured output (existing)
    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None

    # Generation parameters (existing)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # NEW: Multimodal inputs
    multimodal_inputs: Optional[List[MultimodalInput]] = None
    urls: Optional[List[str]] = None  # For URL context tool

    # NEW: Provider-specific features
    tools: Optional[List[Dict]] = None  # Gemini tools like url_context

    def validate(self) -> None:
        """Validate request constraints"""
        # URL validation (Gemini: max 20 URLs)
        if self.urls and len(self.urls) > 20:
            raise ValueError(f"Too many URLs: {len(self.urls)} (max 20)")

        # Validate multimodal inputs
        if self.multimodal_inputs:
            for input_item in self.multimodal_inputs:
                input_item.validate()

        # Image count validation (Gemini: max 3600)
        image_count = sum(
            1 for item in (self.multimodal_inputs or [])
            if item.input_type == MultimodalInputType.IMAGE
        )
        if image_count > 3600:
            raise ValueError(f"Too many images: {image_count} (max 3600)")
```

**Response Metadata**:
```python
@dataclass
class URLContextMetadata:
    """Metadata about URL retrieval from url_context tool"""
    url: str
    status: Literal["success", "failed", "blocked"]  # blocked = safety filter
    error_message: Optional[str] = None

@dataclass
class LLMResponse:
    """Response from LLM with multimodal metadata"""

    # Text output (existing)
    text: str

    # Cost tracking (existing)
    input_tokens: int
    output_tokens: int
    total_cost: float

    # Provider metadata (existing)
    provider: str
    model: str

    # NEW: Multimodal metadata
    url_context_metadata: Optional[List[URLContextMetadata]] = None
    total_pages_processed: Optional[int] = None  # For documents
    total_images_processed: Optional[int] = None
```

---

## 🔧 Implementation Plan

### Phase 1: Foundation & Data Structures (2-3 days)

**Goal**: Create provider-agnostic multimodal abstractions.

**Tasks**:
1. Create `src/mad_spark_alt/core/multimodal.py`
   - `MultimodalInput` dataclass
   - `MultimodalSourceType` and `MultimodalInputType` enums
   - Validation logic for images, documents, URLs

2. Update `src/mad_spark_alt/core/llm_provider.py`
   - Add `multimodal_inputs` and `urls` to `LLMRequest`
   - Add `url_context_metadata` to `LLMResponse`
   - Update validation methods

3. Create `src/mad_spark_alt/utils/multimodal_utils.py`
   - `detect_mime_type(file_path: str) -> str`
   - `read_file_as_base64(file_path: str) -> Tuple[str, str]`
   - `validate_url(url: str) -> bool`
   - `get_file_size(file_path: str) -> int`
   - `get_pdf_page_count(file_path: str) -> int`

**Testing**:
```python
# tests/test_multimodal.py
def test_multimodal_input_validation():
    # Test image validation (size, MIME type)
    # Test document validation (page count, MIME type)
    # Test source type validation

def test_llm_request_multimodal():
    # Test request with images
    # Test request with documents
    # Test request with URLs
    # Test request with mixed inputs

def test_url_validation():
    # Test max 20 URLs
    # Test URL format validation
```

**Success Criteria**:
- ✅ All data structures created with full type hints
- ✅ Validation logic comprehensive (size, count, format)
- ✅ Unit tests for all validation paths
- ✅ Zero impact on existing text-only code paths

---

### Phase 2: Gemini Provider Implementation (3-4 days)

**Goal**: Implement multimodal support in `GoogleProvider`.

**Tasks**:
1. Update `GoogleProvider.generate()` method
   - Translate `MultimodalInput` to Gemini `contents` array
   - Handle images (inline base64, File API)
   - Handle documents (inline, File API)
   - Add URL context tool when `urls` present
   - Parse `url_context_metadata` from response

2. Implement `contents` array builder
   ```python
   def _build_contents(self, request: LLMRequest) -> List[Dict]:
       """
       Build Gemini contents array from multimodal inputs.

       Order (per Gemini best practices):
       1. Images/documents (if single item, before text)
       2. Text prompt

       Returns array of parts for Gemini API.
       """
       contents = []

       # Add multimodal inputs
       if request.multimodal_inputs:
           for item in request.multimodal_inputs:
               part = self._create_multimodal_part(item)
               contents.append(part)

       # Add text prompt (after images per best practices)
       text_prompt = self._build_prompt(request)
       if request.urls:
           # Include URLs in prompt text
           text_prompt += f"\n\nRelevant URLs: {', '.join(request.urls)}"

       contents.append({"text": text_prompt})

       return [{"role": "user", "parts": contents}]
   ```

3. Implement multimodal part creation
   ```python
   def _create_multimodal_part(self, item: MultimodalInput) -> Dict:
       """Create Gemini part from MultimodalInput"""
       if item.source_type == MultimodalSourceType.BASE64:
           return {
               "inline_data": {
                   "mime_type": item.mime_type,
                   "data": item.data
               }
           }
       elif item.source_type == MultimodalSourceType.FILE_PATH:
           # Read file and convert to base64
           base64_data, mime_type = read_file_as_base64(item.data)
           return {
               "inline_data": {
                   "mime_type": mime_type,
                   "data": base64_data
               }
           }
       elif item.source_type == MultimodalSourceType.FILE_API:
           # Reference uploaded file
           return {
               "file_data": {
                   "file_uri": item.data,  # File API ID
                   "mime_type": item.mime_type
               }
           }
       else:
           raise ValueError(f"Unsupported source type: {item.source_type}")
   ```

4. Add URL context tool integration
   ```python
   def _build_payload(self, request: LLMRequest) -> Dict:
       """Build Gemini API payload with optional tools"""
       payload = {
           "contents": self._build_contents(request),
           "generationConfig": self._build_generation_config(request)
       }

       # Add URL context tool if URLs provided
       if request.urls:
           payload["tools"] = [{"url_context": {}}]

       return payload
   ```

5. Parse URL metadata from response
   ```python
   def _parse_response(self, raw_response: Dict, request: LLMRequest) -> LLMResponse:
       """Parse Gemini response including URL metadata"""
       # ... existing parsing ...

       # Parse URL context metadata if present
       url_metadata = None
       if "url_context_metadata" in raw_response:
           url_metadata = [
               URLContextMetadata(
                   url=meta["url"],
                   status=meta["status"],
                   error_message=meta.get("error_message")
               )
               for meta in raw_response["url_context_metadata"]
           ]

       return LLMResponse(
           text=text,
           input_tokens=input_tokens,
           output_tokens=output_tokens,
           total_cost=cost,
           provider="google",
           model=model_name,
           url_context_metadata=url_metadata
       )
   ```

**Testing**:
```python
# tests/test_google_provider_multimodal.py

@pytest.mark.asyncio
async def test_google_provider_with_image(mock_aiohttp):
    """Test GoogleProvider handles image inputs"""
    request = LLMRequest(
        user_prompt="Describe this image",
        multimodal_inputs=[
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.BASE64,
                data="base64encodeddata",
                mime_type="image/png"
            )
        ]
    )

    provider = GoogleProvider(api_key="test")
    # Mock aiohttp response
    # Verify payload structure has correct contents array
    # Verify inline_data format

@pytest.mark.asyncio
async def test_google_provider_with_urls(mock_aiohttp):
    """Test GoogleProvider handles URL context"""
    request = LLMRequest(
        user_prompt="Summarize these sources",
        urls=["https://example.com/article1", "https://example.com/article2"]
    )

    provider = GoogleProvider(api_key="test")
    # Mock response with url_context_metadata
    # Verify tools array contains url_context
    # Verify metadata parsing

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_gemini_with_image():
    """Integration test with real Gemini API"""
    provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))

    # Read test image
    base64_data, mime_type = read_file_as_base64("tests/fixtures/test_image.jpg")

    request = LLMRequest(
        user_prompt="What do you see in this image?",
        multimodal_inputs=[
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.BASE64,
                data=base64_data,
                mime_type=mime_type
            )
        ]
    )

    response = await provider.generate(request)

    assert response.text
    assert response.total_cost > 0
    assert response.total_images_processed == 1
```

**Success Criteria**:
- ✅ GoogleProvider correctly translates MultimodalInput to Gemini format
- ✅ Images work (inline base64, local files)
- ✅ Documents work (PDF processing)
- ✅ URL context tool integration works
- ✅ URL metadata correctly parsed
- ✅ All unit tests pass
- ✅ Integration test with real API passes

---

### Phase 3: QADI Orchestrator Integration (2-3 days)

**Goal**: Enable QADI cycle to accept and pass through multimodal inputs.

**Tasks**:
1. Update `PhaseInput` dataclass
   ```python
   # src/mad_spark_alt/core/phase_logic.py

   @dataclass
   class PhaseInput:
       """Common inputs for all QADI phases"""
       question: str
       llm_manager: LLMManager
       model_config: ModelConfig
       context: Dict[str, Any]

       # NEW: Multimodal inputs
       multimodal_inputs: Optional[List[MultimodalInput]] = None
       urls: Optional[List[str]] = None
   ```

2. Update phase logic functions to accept multimodal inputs
   ```python
   async def execute_abduction_phase(
       phase_input: PhaseInput,
       num_hypotheses: int = 3
   ) -> PhaseResult:
       """Generate hypotheses with optional multimodal context"""

       # Build prompt mentioning visual/document/URL context
       prompt = build_hypothesis_prompt(
           question=phase_input.question,
           num_hypotheses=num_hypotheses,
           has_multimodal=bool(phase_input.multimodal_inputs),
           has_urls=bool(phase_input.urls)
       )

       # Create LLM request with multimodal inputs
       request = LLMRequest(
           user_prompt=prompt,
           multimodal_inputs=phase_input.multimodal_inputs,
           urls=phase_input.urls,
           response_schema=hypothesis_schema,
           response_mime_type="application/json",
           temperature=phase_input.model_config.temperature,
           max_tokens=phase_input.model_config.max_output_tokens
       )

       # Generate with LLM
       response = await phase_input.llm_manager.generate(request)

       # Parse hypotheses (existing logic)
       hypotheses = parse_hypotheses(response.text, num_hypotheses)

       # Track multimodal metadata
       metadata = {
           "images_processed": response.total_images_processed,
           "pages_processed": response.total_pages_processed,
           "url_retrieval": response.url_context_metadata
       }

       return PhaseResult(
           success=True,
           data=hypotheses,
           llm_cost=response.total_cost,
           metadata=metadata,
           errors=[]
       )
   ```

3. Update orchestrators to accept multimodal inputs
   ```python
   # src/mad_spark_alt/core/simple_qadi_orchestrator.py

   class SimpleQADIOrchestrator(BaseOrchestrator):
       async def run_qadi_cycle(
           self,
           user_input: str,
           context: Optional[str] = None,
           # NEW: Multimodal parameters
           multimodal_inputs: Optional[List[MultimodalInput]] = None,
           urls: Optional[List[str]] = None
       ) -> SimpleQADIResult:
           """
           Execute QADI cycle with optional multimodal inputs.

           Args:
               user_input: The question or prompt
               context: Optional additional text context
               multimodal_inputs: Optional images or documents
               urls: Optional URLs for context retrieval
           """
           # Store multimodal inputs in orchestrator state
           self.multimodal_inputs = multimodal_inputs
           self.urls = urls

           # Pass through to phases
           phase_input = PhaseInput(
               question=user_input,
               llm_manager=self.llm_manager,
               model_config=self.model_config,
               context=self.context,
               multimodal_inputs=multimodal_inputs,
               urls=urls
           )

           # Execute phases (existing logic)
           return await self._execute_phases(phase_input)
   ```

4. Update prompt templates for multimodal awareness
   ```python
   # src/mad_spark_alt/prompts/hypothesis_generation.py

   def build_hypothesis_prompt(
       question: str,
       num_hypotheses: int,
       has_multimodal: bool = False,
       has_urls: bool = False
   ) -> str:
       """Build hypothesis generation prompt with multimodal awareness"""

       base_prompt = f"""Generate {num_hypotheses} diverse hypotheses for: {question}"""

       if has_multimodal:
           base_prompt += """

   Consider the provided images/documents in your analysis. Reference specific
   visual elements, data points, or textual content from the materials."""

       if has_urls:
           base_prompt += """

   Consider the content from the provided URLs. Integrate insights from the
   retrieved web pages into your hypotheses."""

       return base_prompt
   ```

**Testing**:
```python
# tests/test_orchestrator_multimodal.py

@pytest.mark.asyncio
async def test_simple_qadi_with_image():
    """Test SimpleQADI with image input"""
    orchestrator = SimpleQADIOrchestrator(...)

    result = await orchestrator.run_qadi_cycle(
        user_input="Analyze this system architecture",
        multimodal_inputs=[
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data="tests/fixtures/architecture.png",
                mime_type="image/png"
            )
        ]
    )

    assert result.hypotheses
    assert len(result.hypotheses) >= 3
    assert result.total_llm_cost > 0

@pytest.mark.asyncio
async def test_simple_qadi_with_document():
    """Test SimpleQADI with PDF document"""
    result = await orchestrator.run_qadi_cycle(
        user_input="Summarize key findings",
        multimodal_inputs=[
            MultimodalInput(
                input_type=MultimodalInputType.DOCUMENT,
                source_type=MultimodalSourceType.FILE_PATH,
                data="tests/fixtures/research_paper.pdf",
                mime_type="application/pdf"
            )
        ]
    )

    assert result.hypotheses
    assert result.total_llm_cost > 0

@pytest.mark.asyncio
async def test_simple_qadi_with_urls():
    """Test SimpleQADI with URL context"""
    result = await orchestrator.run_qadi_cycle(
        user_input="What are the best practices?",
        urls=["https://example.com/best-practices"]
    )

    assert result.hypotheses
    # Verify URL metadata tracked
```

**Success Criteria**:
- ✅ All orchestrators accept multimodal parameters
- ✅ Multimodal inputs passed through all QADI phases
- ✅ Prompts adapted for multimodal context
- ✅ Metadata tracked (images/pages/URLs processed)
- ✅ Integration tests pass with real API

---

### Phase 4: CLI Integration (2-3 days)

**Goal**: Expose multimodal features via CLI with intuitive flags.

**CRITICAL NOTE**: The system has TWO CLI entry points:
- **`mad_spark_alt`** (primary QADI command) - This is what we'll modify
- **`mad-spark`** (evaluation CLI with subcommands) - Not modified for multimodal

We'll add multimodal flags to the main `mad_spark_alt` command by modifying `qadi_simple.py`.

**Tasks**:
1. Add CLI flags to main `mad_spark_alt` command
   ```python
   # qadi_simple.py (which mad_spark_alt delegates to)

   parser.add_argument("user_input", help="Your question or prompt")
   parser.add_argument("--context", "-c", help="Additional text context")

   # NEW: Multimodal input flags
   @click.option(
       "--image", "-i",
       multiple=True,
       type=click.Path(exists=True),
       help="Local image file path(s). Supports PNG, JPEG, WebP, HEIC."
   )
   @click.option(
       "--image-url",
       multiple=True,
       help="Remote image URL(s). Fetched and analyzed by Gemini."
   )
   @click.option(
       "--document", "-d",
       multiple=True,
       type=click.Path(exists=True),
       help="Local document file path(s). PDF recommended (up to 1000 pages)."
   )
   @click.option(
       "--url", "-u",
       multiple=True,
       help="URL(s) for context retrieval (max 20). Supports HTML, PDF, images."
   )

   # Existing flags...
   @click.option("--temperature", type=float, help="LLM temperature")
   @click.option("--num-hypotheses", "-n", type=int, default=3)
   @click.option("--evolve", is_flag=True, help="Enable genetic evolution")

   def qadi(
       user_input: Optional[str],
       context: Optional[str],
       image: Tuple[str, ...],
       image_url: Tuple[str, ...],
       document: Tuple[str, ...],
       url: Tuple[str, ...],
       temperature: Optional[float],
       num_hypotheses: int,
       evolve: bool,
       ...
   ):
       """
       Run QADI analysis with optional multimodal inputs.

       Examples:

         Analyze an architecture diagram:
           uv run mad_spark_alt "Suggest improvements" --image diagram.png

         Compare product screenshots:
           uv run mad_spark_alt "What differentiates these?" --image a.jpg --image b.jpg

         Research paper analysis:
           uv run mad_spark_alt "Summarize methodology" --document paper.pdf

         Web context synthesis:
           uv run mad_spark_alt "Compare approaches" --url https://... --url https://...

         Multi-modal combination:
           uv run mad_spark_alt "Explain this" --image chart.png --url https://context.com
       """
       # Validation
       if not user_input and not image and not image_url and not document:
           console.print("[red]Error: Provide a question or multimodal input[/red]")
           console.print("Try: uv run mad_spark_alt --help")
           return

       # Build multimodal inputs
       multimodal_inputs = []

       # Add local images
       for img_path in image:
           try:
               file_size = get_file_size(img_path)
               mime_type = detect_mime_type(img_path)
               multimodal_inputs.append(MultimodalInput(
                   input_type=MultimodalInputType.IMAGE,
                   source_type=MultimodalSourceType.FILE_PATH,
                   data=img_path,
                   mime_type=mime_type,
                   file_size=file_size
               ))
           except Exception as e:
               console.print(f"[red]Error loading image {img_path}: {e}[/red]")
               return

       # Add remote images
       for img_url_str in image_url:
           try:
               validate_url(img_url_str)
               multimodal_inputs.append(MultimodalInput(
                   input_type=MultimodalInputType.IMAGE,
                   source_type=MultimodalSourceType.URL,
                   data=img_url_str,
                   mime_type="image/jpeg"  # Gemini auto-detects
               ))
           except Exception as e:
               console.print(f"[red]Error with image URL {img_url_str}: {e}[/red]")
               return

       # Add documents
       for doc_path in document:
           try:
               file_size = get_file_size(doc_path)
               mime_type = detect_mime_type(doc_path)
               page_count = None
               if mime_type == "application/pdf":
                   page_count = get_pdf_page_count(doc_path)
                   if page_count > 1000:
                       console.print(f"[yellow]Warning: {doc_path} has {page_count} pages (max 1000)[/yellow]")

               multimodal_inputs.append(MultimodalInput(
                   input_type=MultimodalInputType.DOCUMENT,
                   source_type=MultimodalSourceType.FILE_PATH,
                   data=doc_path,
                   mime_type=mime_type,
                   file_size=file_size,
                   page_count=page_count
               ))
           except Exception as e:
               console.print(f"[red]Error loading document {doc_path}: {e}[/red]")
               return

       # Validate URLs
       urls_list = list(url) if url else None
       if urls_list:
           if len(urls_list) > 20:
               console.print(f"[red]Error: Too many URLs ({len(urls_list)}). Max 20.[/red]")
               return

           for url_str in urls_list:
               try:
                   validate_url(url_str)
               except Exception as e:
                   console.print(f"[red]Invalid URL {url_str}: {e}[/red]")
                   return

       # Display multimodal inputs summary
       if multimodal_inputs or urls_list:
           console.print("\n[bold cyan]📎 Multimodal Inputs[/bold cyan]")
           if multimodal_inputs:
               for item in multimodal_inputs:
                   icon = "🖼️" if item.input_type == MultimodalInputType.IMAGE else "📄"
                   name = Path(item.data).name if item.source_type == MultimodalSourceType.FILE_PATH else item.data
                   size_str = f" ({item.file_size // 1024} KB)" if item.file_size else ""
                   pages_str = f" ({item.page_count} pages)" if item.page_count else ""
                   console.print(f"  {icon} {name}{size_str}{pages_str}")

           if urls_list:
               console.print(f"  🔗 {len(urls_list)} URL(s) for context")
           console.print()

       # Setup LLM provider
       setup_result = await setup_llm_providers(
           google_api_key=os.getenv("GOOGLE_API_KEY")
       )
       if not setup_result:
           console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
           return

       # Create orchestrator
       orchestrator = SimpleQADIOrchestrator(
           temperature_override=temperature,
           num_hypotheses=num_hypotheses
       )

       # Run QADI cycle with multimodal inputs
       try:
           result = await orchestrator.run_qadi_cycle(
               user_input=user_input or "Analyze the provided materials",
               context=context,
               multimodal_inputs=multimodal_inputs if multimodal_inputs else None,
               urls=urls_list
           )

           # Display results (existing display logic)
           display_qadi_result(result)

           # Display multimodal metadata
           if result.metadata:
               console.print("\n[bold cyan]📊 Processing Stats[/bold cyan]")
               if result.metadata.get("images_processed"):
                   console.print(f"  Images: {result.metadata['images_processed']}")
               if result.metadata.get("pages_processed"):
                   console.print(f"  Pages: {result.metadata['pages_processed']}")
               if result.metadata.get("url_retrieval"):
                   success_count = sum(
                       1 for meta in result.metadata["url_retrieval"]
                       if meta.status == "success"
                   )
                   console.print(f"  URLs: {success_count}/{len(urls_list)} retrieved successfully")

       except Exception as e:
           console.print(f"[red]Error during QADI cycle: {e}[/red]")
           logger.exception("QADI cycle failed")
           return
   ```

2. Add file path resolution and validation utilities
   ```python
   # src/mad_spark_alt/utils/multimodal_utils.py

   from pathlib import Path
   import os

   def resolve_file_path(file_path: str) -> Path:
       """
       Resolve and validate file path with proper expansion.

       Supports:
       - Relative paths (./file.png, ../file.png, file.png)
       - Absolute paths (/full/path/to/file.png)
       - Home directory (~/ or ~user/)
       - Environment variables ($HOME/file.png)

       Args:
           file_path: Path string from user input

       Returns:
           Resolved absolute Path object

       Raises:
           FileNotFoundError: If file doesn't exist
           ValueError: If path is a directory, not a file
       """
       # Expand ~ and environment variables
       expanded = os.path.expanduser(os.path.expandvars(file_path))

       # Convert to Path and resolve to absolute
       path = Path(expanded).resolve()

       # Validate existence
       if not path.exists():
           raise FileNotFoundError(
               f"File not found: {file_path}\n"
               f"Resolved to: {path}\n"
               f"Current directory: {Path.cwd()}"
           )

       # Validate it's a file (not directory)
       if not path.is_file():
           raise ValueError(f"Path is not a file: {path}")

       return path

   def validate_file_path(path: Path, max_size_mb: int = 50) -> None:
       """
       Validate file is safe to process.

       Args:
           path: Path object to validate
           max_size_mb: Maximum file size in MB (default: 50MB)

       Raises:
           ValueError: If file is too large
           PermissionError: If file is not readable
       """
       # Size check (prevent huge files)
       size_mb = path.stat().st_size / (1024 * 1024)
       if size_mb > max_size_mb:
           raise ValueError(
               f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)\n"
               f"For files >20MB, consider using Gemini File API"
           )

       # Readable check
       if not os.access(path, os.R_OK):
           raise PermissionError(f"Cannot read file: {path}")

   def get_pdf_page_count(file_path: Path) -> Optional[int]:
       """Get page count from PDF file"""
       try:
           import PyPDF2
           with open(file_path, "rb") as f:
               reader = PyPDF2.PdfReader(f)
               return len(reader.pages)
       except ImportError:
           logger.warning("PyPDF2 not installed, cannot count PDF pages")
           return None
       except Exception as e:
           logger.warning(f"Error counting PDF pages: {e}")
           return None
   ```

3. Update CLI help and examples with path flexibility
   ```python
   # Add to qadi_simple.py help text
   """
   Multimodal Examples:

     File paths can be:
     - Relative: diagram.png, ./images/chart.png, ../docs/paper.pdf
     - Absolute: /Users/name/Downloads/file.png
     - Home dir: ~/Desktop/screenshot.png, ~/Documents/research.pdf

     1. Image Analysis (relative path)
        uv run mad_spark_alt "What architectural patterns are used?" \\
          --image diagram.png

     2. Image Analysis (home directory)
        uv run mad_spark_alt "Analyze this screenshot" \\
          --image ~/Desktop/Screenshot.png

     3. Document Processing (Downloads folder)
        uv run mad_spark_alt "Summarize key findings" \\
          --document ~/Downloads/research_paper.pdf

     4. Multi-Modal Combination (mixed paths)
        uv run mad_spark_alt "Analyze competitive landscape" \\
          --image ~/Desktop/our_product.png \\
          --image-url https://competitor.com/screenshot.png \\
          --document ./analysis/market_report.pdf \\
          --url https://competitor.com/pricing

     5. Project-Organized Files (relative paths)
        cd my_project
        uv run mad_spark_alt "Compare designs" \\
          --image designs/option_a.png \\
          --image designs/option_b.png \\
          --document specs/requirements.pdf

     6. Evolution with Images
        uv run mad_spark_alt "Improve this design" \\
          --image ~/Documents/current_design.png \\
          --evolve --generations 3 --population 5
   """
   ```

**Testing**:
```bash
# Manual CLI testing - Test various path types

# Test 1: Relative path (current directory)
uv run mad_spark_alt "Describe this architecture" \
  --image tests/fixtures/architecture.png

# Test 2: Absolute path
uv run mad_spark_alt "Analyze this screenshot" \
  --image /tmp/test_screenshot.png

# Test 3: Home directory expansion
uv run mad_spark_alt "What's in this image?" \
  --image ~/Downloads/diagram.png

# Test 4: Multiple images from different locations
uv run mad_spark_alt "Compare these designs" \
  --image ./design_a.png \
  --image ~/Desktop/design_b.png \
  --image /tmp/design_c.png

# Test 5: PDF document (Downloads folder - common use case)
uv run mad_spark_alt "Summarize key points" \
  --document ~/Downloads/research_paper.pdf

# Test 6: URL context only
uv run mad_spark_alt "What are common patterns?" \
  --url https://example.com/article

# Test 7: Multi-modal combination (mixed paths)
uv run mad_spark_alt "Analyze from all sources" \
  --image ~/Desktop/chart.png \
  --document ./reports/analysis.pdf \
  --url https://context.com \
  --context "Focus on 2024 trends"

# Test 8: Evolution with image (home directory)
uv run mad_spark_alt "Improve this UI" \
  --image ~/Documents/current_ui.png \
  --evolve --generations 2 --population 3

# Test 9: File not found error handling
uv run mad_spark_alt "Test error" \
  --image nonexistent.png
# Expected: Clear error message with current directory info

# Test 10: Directory instead of file error
uv run mad_spark_alt "Test error" \
  --image ./tests/
# Expected: Error message indicating path is a directory
```

**Success Criteria**:
- ✅ All CLI flags working
- ✅ Input validation comprehensive
- ✅ Error messages helpful
- ✅ Progress indicators for large files
- ✅ Metadata display clear and informative
- ✅ Help text with examples

---

### Phase 5: Documentation & Polish (1-2 days)

**Goal**: Comprehensive documentation and examples.

**Tasks**:
1. Update README.md with multimodal examples
2. Create `docs/MULTIMODAL.md` with detailed guide
3. Update ARCHITECTURE.md with new components
4. Create example notebooks (Jupyter)
5. Add cost estimation guide for multimodal inputs
6. Document provider limitations (Gemini-specific features)

**Documentation Structure**:
```markdown
# docs/MULTIMODAL.md

## Multimodal Capabilities

### Supported Input Types
- Images (PNG, JPEG, WebP, HEIC, HEIF)
- Documents (PDF with vision understanding)
- URLs (HTML, PDF, images - up to 20 per request)

### Usage Examples

#### Python API
[Code examples...]

#### CLI
[Command examples...]

### Cost Estimation

| Input Type | Tokens | Approx Cost (Gemini 2.5 Flash) |
|------------|--------|--------------------------------|
| Image ≤384px | 258 | $0.0000193 |
| Image >384px (per tile) | 258 | $0.0000193 |
| PDF Page | 258 | $0.0000193 |
| URL (varies by size) | Actual content | Varies |

### Limitations

#### Gemini-Specific
- Max 20 URLs per request
- Max 34MB per URL
- Max 1000 PDF pages
- URL context doesn't support paywalled content

#### Provider Lock-in
These features currently require Gemini API. OpenAI and Anthropic
have different interfaces for multimodal inputs.

### Migration Guide

If you need to switch providers in the future, refer to
the `MultimodalInput` abstraction which provides a
provider-agnostic interface.
```

**Success Criteria**:
- ✅ README updated with multimodal section
- ✅ Complete MULTIMODAL.md guide
- ✅ ARCHITECTURE.md reflects new components
- ✅ Example notebooks with real use cases
- ✅ Cost guide accurate
- ✅ Provider limitations clearly documented

---

## ⚠️ Provider Lock-in Analysis

### Current State

**Gemini-Dependent Features**:
1. **URL Context Tool** - Gemini-specific `url_context` tool
2. **File API** - Gemini's 50MB, 48hr file storage
3. **PDF Vision** - Native document understanding (1000 pages)
4. **Contents Array** - Gemini's part-based multimodal structure

### Multi-Provider Comparison

| Feature | Gemini | OpenAI GPT-4V | Anthropic Claude 3.5 |
|---------|--------|---------------|----------------------|
| **API Access** | `google.generativeai` | `openai` | `anthropic` |
| **Image Input** | `inline_data` or `file_data` | `image_url` or base64 | base64 in `content` array |
| **PDF Support** | Native vision (1000 pages) | Text extraction + images | Text extraction only |
| **URL Fetching** | Native `url_context` tool | Manual fetch required | Manual fetch required |
| **File Storage** | File API (50MB, 48hr) | None (inline only) | None (inline only) |
| **Max Image Size** | 20MB inline, unlimited via File API | 20MB (varies) | 5MB (varies) |
| **Request Format** | `contents` array with `parts` | `messages` with `content` blocks | `messages` with `content` blocks |

### Migration Strategy for Multi-Provider

**Option 1: Gemini-Only (Current Plan)**
- ✅ Fastest implementation
- ✅ Best multimodal support
- ❌ Vendor lock-in

**Option 2: Provider Abstraction Layer**
- ✅ Future multi-provider support
- ❌ Significant additional complexity
- ❌ Feature parity challenges (URL tool doesn't exist in others)

**Option 3: Hybrid Approach**
- Implement Gemini-specific features now
- Design abstraction layer for basic features (images only)
- Document provider-specific capabilities
- Allow manual URL fetching for non-Gemini providers

### Recommendation

**Accept Gemini lock-in for v1.0**, with these safeguards:

1. **Abstraction Layer**: Use `MultimodalInput` as provider-agnostic format
2. **Documentation**: Clearly mark Gemini-specific features
3. **Feature Flags**: Design system to gracefully degrade on other providers
   - Images: Work on all providers (different encoding)
   - Documents: Gemini-only (others require preprocessing)
   - URLs: Gemini-only (others need manual fetch)
4. **Future Path**: Provider-specific implementations can translate `MultimodalInput`

**Migration Example** (Future OpenAI Support):
```python
class OpenAIProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Translate MultimodalInput to OpenAI format
        messages = []

        for item in request.multimodal_inputs or []:
            if item.input_type == MultimodalInputType.IMAGE:
                if item.source_type == MultimodalSourceType.BASE64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
                        ]
                    })
                else:
                    # Convert file_path to base64
                    ...
            elif item.input_type == MultimodalInputType.DOCUMENT:
                # OpenAI doesn't support PDF vision
                raise NotImplementedError("PDF vision not supported by OpenAI")

        # URLs require manual fetch for OpenAI
        if request.urls:
            # Fetch URLs manually, add as text context
            for url in request.urls:
                content = await fetch_url(url)
                messages.append({"role": "user", "content": content})

        # Rest of OpenAI-specific logic...
```

---

## 📊 Cost Implications

### Token Cost Calculation

**Images**:
- Small (≤384px): 258 tokens
- Medium/Large: 258 tokens per 768×768 tile
- Example: 1920×1080 image ≈ 3 tiles ≈ 774 tokens

**Documents**:
- ~258 tokens per PDF page
- Example: 50-page report ≈ 12,900 tokens

**URLs**:
- Varies by content size
- HTML page: 1,000-10,000 tokens
- PDF from URL: Same as document (258 tokens/page)

### Cost Examples (Gemini 2.5 Flash)

| Use Case | Input Tokens | Output Tokens | Est. Cost |
|----------|--------------|---------------|-----------|
| 1 image + analysis | ~500 | ~1000 | $0.00011 |
| 3 images + comparison | ~1500 | ~1500 | $0.00023 |
| 50-page PDF summary | ~13,000 | ~2000 | $0.00112 |
| 5 URLs + synthesis | ~25,000 | ~2000 | $0.00202 |
| Multi-modal (2 images + 1 PDF + 2 URLs) | ~30,000 | ~3000 | $0.00247 |

**Notes**:
- Gemini 2.5 Flash: $0.075/1M input tokens, $0.30/1M output tokens
- Costs are approximate, vary with content
- URL content size can vary significantly

---

## 🚧 Known Limitations & Constraints

### Technical Constraints

1. **URL Context**
   - Max 20 URLs per request
   - Max 34MB per URL
   - No paywalled content
   - No YouTube videos
   - Content counts toward token limit

2. **Images**
   - Max 3,600 images per request (Gemini)
   - 20MB limit for inline encoding
   - Must use File API for larger files

3. **Documents**
   - Max 1,000 PDF pages per request
   - PDF only for vision understanding
   - Other formats (TXT, HTML) are text-extracted only

4. **General**
   - All multimodal content counts toward token limits
   - Large inputs significantly increase costs
   - Processing time increases with content size

### Provider-Specific

**Gemini Only**:
- Native URL fetching (url_context tool)
- PDF vision with diagrams/charts/tables
- File API for large files (50MB, 48hr retention)
- Bounding box detection (Gemini 2.0+)
- Segmentation masks (Gemini 2.5+)

**OpenAI Limitations** (if future support added):
- Manual URL fetching required
- No native PDF vision (text extraction only)
- No File API (inline base64 only)

**Anthropic Limitations** (if future support added):
- No URL fetching
- No PDF vision
- Smaller file size limits

---

## 🎯 Success Metrics

### Phase Completion Criteria

**Phase 1: Foundation**
- ✅ `MultimodalInput` dataclass complete with validation
- ✅ `LLMRequest` extended with multimodal fields
- ✅ Utility functions for MIME detection, base64 encoding
- ✅ 90%+ test coverage on validation logic

**Phase 2: Provider**
- ✅ GoogleProvider handles all multimodal input types
- ✅ URL context tool integration working
- ✅ Metadata parsing (url_context_metadata)
- ✅ Integration tests pass with real Gemini API

**Phase 3: Orchestrator**
- ✅ PhaseInput accepts multimodal parameters
- ✅ All QADI phases support multimodal context
- ✅ Prompts adapted for visual/document awareness
- ✅ Metadata tracked and returned

**Phase 4: CLI**
- ✅ All flags (--image, --document, --url) working
- ✅ Input validation comprehensive
- ✅ Error messages actionable
- ✅ Display shows multimodal processing stats

**Phase 5: Documentation**
- ✅ README updated with examples
- ✅ MULTIMODAL.md guide complete
- ✅ Example notebooks created
- ✅ Cost guide accurate
- ✅ Provider limitations documented

### Overall Success

**Functional**:
- Users can analyze images via CLI
- Users can process PDFs via CLI
- Users can fetch URL context via CLI
- Multi-modal combinations work seamlessly
- Evolution system works with multimodal inputs

**Quality**:
- Zero regressions in text-only paths
- 85%+ test coverage for multimodal code
- <100ms overhead for multimodal input processing
- Clear error messages for all failure modes

**User Experience**:
- CLI examples in help text
- Progress indicators for large files
- Cost estimates shown before processing
- Multimodal metadata displayed after processing

---

## 📝 Next Steps

1. **Get approval** on this plan
2. **Create feature branch**: `git checkout -b feature/multimodal-support`
3. **Start Phase 1**: Foundation & data structures
4. **Daily progress updates** in session handover docs
5. **TDD approach**: Write tests before implementation
6. **Integration tests early**: Test with real Gemini API in Phase 2

---

## 📚 References

### API Documentation
- [Gemini URL Context](https://ai.google.dev/gemini-api/docs/url-context)
- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Gemini Document Processing](https://ai.google.dev/gemini-api/docs/document-processing)

### Related Files
- `src/mad_spark_alt/core/llm_provider.py` - LLM request/response interfaces
- `src/mad_spark_alt/core/phase_logic.py` - QADI phase logic
- `src/mad_spark_alt/core/simple_qadi_orchestrator.py` - Main orchestrator
- `src/mad_spark_alt/cli.py` - CLI interface

### Project Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines
- [refactoring_plan_20251106.md](refactoring_plan_20251106.md) - Recent refactoring

---

**End of Multimodal Features Plan**
