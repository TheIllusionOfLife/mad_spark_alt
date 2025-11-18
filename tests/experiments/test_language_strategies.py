"""Experimental tests for language mirroring strategies.

This script tests three language instruction strategies across three languages
with multiple runs to determine which approach works best.

Test Matrix:
- 3 strategies (Strategy 1, Strategy 2, Combined)
- 3 languages (English, Japanese, Spanish)
- 5 runs per combination
- Total: 45 tests (run in parallel)
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from mad_spark_alt.core.language_utils import (
    detect_language,
    get_combined_instruction,
    get_strategy_1_instruction,
    get_strategy_2_instruction,
)
from mad_spark_alt.core.llm_provider import LLMRequest, OllamaProvider


# Test prompts in each language
TEST_PROMPTS = {
    "en": "How can we reduce food waste in cities?",
    "ja": "都市部で食品廃棄物を減らすにはどうすればよいですか？",
    "es": "¿Cómo podemos reducir el desperdicio de alimentos en las ciudades?",
}


@dataclass
class ExperimentResult:
    """Result of a single language mirroring experiment."""

    strategy: str
    input_language: str
    run_number: int
    prompt: str
    response: str
    detected_language: str
    success: bool  # True if output language matches input language
    response_time: float
    timestamp: str


class LanguageMirroringExperiment:
    """Experimental framework for testing language mirroring strategies."""

    def __init__(self):
        self.results: List[ExperimentResult] = []

    def get_system_prompt_for_strategy(self, strategy: str) -> str:
        """Get system prompt with language instruction for given strategy.

        Args:
            strategy: "strategy1", "strategy2", or "combined"

        Returns:
            System prompt with language instruction
        """
        if strategy == "strategy1":
            return get_strategy_1_instruction()
        elif strategy == "strategy2":
            return get_strategy_2_instruction()
        elif strategy == "combined":
            return get_combined_instruction()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def run_single_test(
        self,
        strategy: str,
        language: str,
        prompt: str,
        run_num: int
    ) -> ExperimentResult:
        """Run a single language mirroring test.

        Args:
            strategy: Which instruction strategy to use
            language: Expected language code (en, ja, es)
            prompt: Test prompt in the target language
            run_num: Run number (1-5)

        Returns:
            ExperimentResult with test outcome
        """
        provider = OllamaProvider()

        try:
            # Get system prompt with language instruction
            system_prompt = self.get_system_prompt_for_strategy(strategy)

            # Create request
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )

            # Time the request
            start_time = time.time()
            response_obj = await provider.generate(request)
            response_time = time.time() - start_time

            # Detect output language
            detected_lang = detect_language(response_obj.content)
            success = detected_lang == language

            result = ExperimentResult(
                strategy=strategy,
                input_language=language,
                run_number=run_num,
                prompt=prompt,
                response=response_obj.content,
                detected_language=detected_lang,
                success=success,
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )

            self.results.append(result)
            return result

        finally:
            await provider.close()

    async def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all 45 experiments in parallel.

        Returns:
            List of all experiment results
        """
        tasks = []

        strategies = ["strategy1", "strategy2", "combined"]
        runs_per_combo = 5

        for strategy in strategies:
            for lang_code, prompt in TEST_PROMPTS.items():
                for run_num in range(1, runs_per_combo + 1):
                    task = self.run_single_test(
                        strategy=strategy,
                        language=lang_code,
                        prompt=prompt,
                        run_num=run_num
                    )
                    tasks.append(task)

        print(f"\nStarting {len(tasks)} language mirroring experiments in parallel...")
        print("This may take 10-15 minutes with Ollama...")

        results = await asyncio.gather(*tasks)

        print(f"✅ All {len(results)} experiments completed!")
        return results

    def analyze_results(self) -> Dict:
        """Analyze experiment results and generate report.

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_tests": len(self.results),
            "by_strategy": {},
            "by_language": {},
            "overall_success_rate": 0.0,
            "recommendation": "",
        }

        # Calculate success rates by strategy
        for strategy in ["strategy1", "strategy2", "combined"]:
            strategy_results = [r for r in self.results if r.strategy == strategy]
            if strategy_results:
                success_count = sum(1 for r in strategy_results if r.success)
                total = len(strategy_results)
                success_rate = (success_count / total) * 100

                analysis["by_strategy"][strategy] = {
                    "total": total,
                    "successes": success_count,
                    "failures": total - success_count,
                    "success_rate": success_rate,
                    "avg_response_time": sum(r.response_time for r in strategy_results) / total,
                }

        # Calculate success rates by language
        for lang in ["en", "ja", "es"]:
            lang_results = [r for r in self.results if r.input_language == lang]
            if lang_results:
                success_count = sum(1 for r in lang_results if r.success)
                total = len(lang_results)
                success_rate = (success_count / total) * 100

                analysis["by_language"][lang] = {
                    "total": total,
                    "successes": success_count,
                    "failures": total - success_count,
                    "success_rate": success_rate,
                }

        # Determine overall success rate
        total_successes = sum(1 for r in self.results if r.success)
        analysis["overall_success_rate"] = (total_successes / len(self.results)) * 100

        # Determine recommendation (strategy with highest success rate)
        best_strategy = max(
            analysis["by_strategy"].items(),
            key=lambda x: x[1]["success_rate"]
        )
        analysis["recommendation"] = best_strategy[0]

        return analysis

    def print_report(self, analysis: Dict):
        """Print formatted analysis report.

        Args:
            analysis: Analysis results from analyze_results()
        """
        print("\n" + "=" * 80)
        print("Language Mirroring Experiment Results")
        print("=" * 80)

        print(f"\nTest Configuration:")
        print(f"  Strategies: 3 (Strategy 1, Strategy 2, Combined)")
        print(f"  Languages: 3 (English, Japanese, Spanish)")
        print(f"  Runs per combination: 5")
        print(f"  Total tests: {analysis['total_tests']}")
        print(f"  Model: gemma3:12b-it-qat (Ollama)")

        print(f"\nOverall Results:")
        print(f"  Success Rate: {analysis['overall_success_rate']:.1f}%")

        print(f"\nResults by Strategy:")
        for strategy, stats in analysis["by_strategy"].items():
            print(f"\n{strategy.upper()}:")
            print(f"  Total runs: {stats['total']}")
            print(f"  Successes: {stats['successes']}")
            print(f"  Failures: {stats['failures']}")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"  Avg response time: {stats['avg_response_time']:.2f}s")

            # Show per-language breakdown
            strategy_results = [r for r in self.results if r.strategy == strategy]
            for lang in ["en", "ja", "es"]:
                lang_results = [r for r in strategy_results if r.input_language == lang]
                if lang_results:
                    successes = sum(1 for r in lang_results if r.success)
                    total = len(lang_results)
                    print(f"    {lang.upper()}: {successes}/{total} ({(successes/total)*100:.0f}%)")

        print(f"\nResults by Language:")
        for lang, stats in analysis["by_language"].items():
            symbol = "✓" if stats["success_rate"] >= 80 else "⚠"
            print(f"  {lang.upper()}: {stats['successes']}/{stats['total']} ({stats['success_rate']:.1f}%) {symbol}")

        print(f"\n{'='*80}")
        print(f"🏆 RECOMMENDATION: Use {analysis['recommendation'].upper()}")
        print(f"   (Best success rate: {analysis['by_strategy'][analysis['recommendation']]['success_rate']:.1f}%)")
        print(f"{'='*80}\n")

    def save_results(self, filepath: str = "experiments/language_strategy_results.json"):
        """Save detailed results to JSON file.

        Args:
            filepath: Path to save results (relative to project root)
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment": "language_mirroring_strategies",
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"📊 Detailed results saved to: {output_path}")


@pytest.mark.ollama
@pytest.mark.integration
@pytest.mark.asyncio
async def test_language_mirroring_strategies():
    """Run comprehensive language mirroring strategy experiments.

    This test runs 45 experiments (3 strategies × 3 languages × 5 runs)
    in parallel to determine which language instruction strategy works best.

    Requires:
    - Ollama server running locally (localhost:11434)
    - gemma3:12b-it-qat model available

    Expected duration: 10-15 minutes (parallel execution)
    """
    experiment = LanguageMirroringExperiment()

    # Run all experiments in parallel
    await experiment.run_all_experiments()

    # Analyze results
    analysis = experiment.analyze_results()

    # Print report
    experiment.print_report(analysis)

    # Save detailed results
    experiment.save_results()

    # Assert minimum success rate
    assert analysis["overall_success_rate"] >= 60.0, \
        f"Overall success rate {analysis['overall_success_rate']:.1f}% is below 60% threshold"

    print(f"\n✅ Experiment complete! Recommended strategy: {analysis['recommendation']}")
