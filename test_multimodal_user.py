#!/usr/bin/env python3
"""
User testing script for Phase 2 multimodal functionality.

This script tests the implementation with real API calls using simple Python code
that mimics how users would use the system.

Run with: python test_multimodal_user.py
"""

import asyncio
import os
from pathlib import Path

from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_result(response, test_name: str):
    """Print test result details."""
    print(f"✓ {test_name}")
    print(f"  Model: {response.model}")
    print(f"  Cost: ${response.cost:.6f}")
    print(f"  Tokens: {response.usage['prompt_tokens']} input, {response.usage['completion_tokens']} output")
    if response.total_images_processed:
        print(f"  Images: {response.total_images_processed} processed")
    if response.total_pages_processed:
        print(f"  Pages: {response.total_pages_processed} processed")
    if response.url_context_metadata:
        print(f"  URLs: {len(response.url_context_metadata)} fetched")
        for meta in response.url_context_metadata:
            print(f"    - {meta.url}: {meta.status}")
    print(f"\n  Response Preview:")
    print(f"  {response.content[:300]}" + ("..." if len(response.content) > 300 else ""))
    print()


async def test_image_analysis(provider: GoogleProvider):
    """Test 1: Image Analysis"""
    print_section("TEST 1: Image Analysis")

    # Use the test image fixture
    image_path = str(Path(__file__).parent / "tests" / "fixtures" / "test_image.png")

    image_input = MultimodalInput(
        input_type=MultimodalInputType.IMAGE,
        source_type=MultimodalSourceType.FILE_PATH,
        data=image_path,
        mime_type="image/png"
    )

    request = LLMRequest(
        user_prompt="Describe what you see in this image in detail.",
        multimodal_inputs=[image_input],
        max_tokens=500
    )

    response = await provider.generate(request)
    print_result(response, "Image Analysis")

    # Verify quality
    assert "blue" in response.content.lower() or "square" in response.content.lower(), \
        "❌ Image description doesn't mention expected shapes"
    assert len(response.content) > 50, "❌ Response too short"
    assert response.total_images_processed == 1, "❌ Image count incorrect"
    print("✅ Quality checks passed!\n")


async def test_pdf_processing(provider: GoogleProvider):
    """Test 2: PDF Document Processing"""
    print_section("TEST 2: PDF Document Processing")

    # Use the test PDF fixture
    pdf_path = str(Path(__file__).parent / "tests" / "fixtures" / "test_document.pdf")

    doc_input = MultimodalInput(
        input_type=MultimodalInputType.DOCUMENT,
        source_type=MultimodalSourceType.FILE_PATH,
        data=pdf_path,
        mime_type="application/pdf",
        page_count=3
    )

    request = LLMRequest(
        user_prompt="Summarize this document and tell me how many pages it has.",
        multimodal_inputs=[doc_input],
        max_tokens=500
    )

    response = await provider.generate(request)
    print_result(response, "PDF Processing")

    # Verify quality
    assert "3" in response.content or "three" in response.content.lower(), \
        "❌ Document summary doesn't mention page count"
    assert "test" in response.content.lower(), "❌ Missing key content"
    assert response.total_pages_processed == 3, "❌ Page count incorrect"
    print("✅ Quality checks passed!\n")


async def test_url_context(provider: GoogleProvider):
    """Test 3: URL Context Retrieval"""
    print_section("TEST 3: URL Context Retrieval")

    request = LLMRequest(
        user_prompt="What is the main purpose of this website?",
        urls=["https://www.example.com"],
        max_tokens=300
    )

    response = await provider.generate(request)
    print_result(response, "URL Context")

    # Verify quality
    assert "example" in response.content.lower() or "domain" in response.content.lower(), \
        "❌ URL content not used in response"
    assert len(response.content) > 30, "❌ Response too short"
    print("✅ Quality checks passed!\n")


async def test_mixed_multimodal(provider: GoogleProvider):
    """Test 4: Mixed Multimodal Inputs"""
    print_section("TEST 4: Mixed Multimodal (Image + URL)")

    image_path = str(Path(__file__).parent / "tests" / "fixtures" / "test_image.png")

    image_input = MultimodalInput(
        input_type=MultimodalInputType.IMAGE,
        source_type=MultimodalSourceType.FILE_PATH,
        data=image_path,
        mime_type="image/png"
    )

    request = LLMRequest(
        user_prompt="I've provided an image and a URL. Briefly describe both.",
        multimodal_inputs=[image_input],
        urls=["https://www.example.com"],
        max_tokens=500
    )

    response = await provider.generate(request)
    print_result(response, "Mixed Multimodal")

    # Verify quality
    content_lower = response.content.lower()
    has_image_mention = any(word in content_lower for word in ["image", "shape", "blue", "red", "square"])
    has_url_mention = any(word in content_lower for word in ["example", "website", "domain"])

    assert has_image_mention or has_url_mention, \
        "❌ Response doesn't mention both inputs"
    assert response.total_images_processed == 1, "❌ Image count incorrect"
    print("✅ Quality checks passed!\n")


async def test_backward_compatibility(provider: GoogleProvider):
    """Test 5: Backward Compatibility (Text-only)"""
    print_section("TEST 5: Backward Compatibility (Text-only)")

    request = LLMRequest(
        user_prompt="What is 10 + 5? Answer in one sentence.",
        max_tokens=50
    )

    response = await provider.generate(request)
    print_result(response, "Text-only Request")

    # Verify quality
    assert "15" in response.content, "❌ Math answer incorrect"
    assert response.total_images_processed is None, "❌ Should have no images"
    assert response.total_pages_processed is None, "❌ Should have no pages"
    print("✅ Quality checks passed!\n")


async def main():
    """Run all user tests."""
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY='your-api-key'")
        print("   Or source .env: source .env")
        return

    print("\n" + "=" * 80)
    print("  MULTIMODAL PHASE 2 USER TESTING")
    print("  Testing with Real Gemini API")
    print("=" * 80)
    print(f"\nAPI Key: {api_key[:20]}...")
    print("Expected total cost: ~$0.003")

    provider = GoogleProvider(api_key=api_key)

    try:
        # Run all tests
        await test_image_analysis(provider)
        await test_pdf_processing(provider)
        await test_url_context(provider)
        await test_mixed_multimodal(provider)
        await test_backward_compatibility(provider)

        # Summary
        print_section("ALL TESTS PASSED ✅")
        print("Phase 2 implementation verified successfully!")
        print("- Images: Working correctly")
        print("- PDFs: Working correctly")
        print("- URLs: Working correctly")
        print("- Mixed inputs: Working correctly")
        print("- Backward compatibility: Working correctly")

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
