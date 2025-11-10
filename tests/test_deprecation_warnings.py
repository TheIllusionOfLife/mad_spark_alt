"""
Tests for deprecation warning behavior.

This test suite ensures that:
1. Module-level imports of core do NOT trigger warnings
2. Explicit imports of deprecated classes/modules DO trigger warnings
3. Warnings contain appropriate messages and stack levels
"""

import sys
import warnings
from io import StringIO

import pytest


class TestDeprecationWarningBehavior:
    """Test that deprecation warnings only fire on explicit use."""

    def test_import_core_no_warnings(self):
        """
        Test that importing mad_spark_alt.core does NOT trigger warnings.

        This is the critical fix - users should be able to import core
        without seeing deprecation warnings for modules they're not using.
        """
        # Clear any previously imported modules
        modules_to_clear = [
            m for m in sys.modules.keys()
            if m.startswith('mad_spark_alt.core')
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import core - should NOT trigger warnings
            import mad_spark_alt.core  # noqa: F401

            # Filter to deprecation warnings only
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]

            # Should have NO deprecation warnings
            assert len(deprecation_warnings) == 0, (
                f"Expected no warnings on core import, but got {len(deprecation_warnings)}: "
                f"{[str(warning.message) for warning in deprecation_warnings]}"
            )

    def test_smart_orchestrator_import_triggers_warning(self):
        """
        Test that explicitly importing SmartQADIOrchestrator triggers a warning.
        """
        # Clear module cache
        if 'mad_spark_alt.core.smart_orchestrator' in sys.modules:
            del sys.modules['mad_spark_alt.core.smart_orchestrator']
        if 'mad_spark_alt.core' in sys.modules:
            del sys.modules['mad_spark_alt.core']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Explicit import should trigger warning
            from mad_spark_alt.core import SmartQADIOrchestrator  # noqa: F401

            # Should have exactly 1 deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'SmartQADIOrchestrator' in str(warning.message)
            ]

            assert len(deprecation_warnings) == 1, (
                f"Expected 1 SmartQADIOrchestrator warning, got {len(deprecation_warnings)}"
            )

            # Verify warning message content
            warning_msg = str(deprecation_warnings[0].message)
            assert 'deprecated' in warning_msg.lower()
            assert 'v2.0.0' in warning_msg
            assert 'UnifiedQADIOrchestrator' in warning_msg

    def test_smart_cycle_result_import_triggers_warning(self):
        """
        Test that importing SmartQADICycleResult triggers a warning.
        """
        # Clear module cache
        if 'mad_spark_alt.core.smart_orchestrator' in sys.modules:
            del sys.modules['mad_spark_alt.core.smart_orchestrator']
        if 'mad_spark_alt.core' in sys.modules:
            del sys.modules['mad_spark_alt.core']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from mad_spark_alt.core import SmartQADICycleResult  # noqa: F401

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'Smart' in str(warning.message)
            ]

            assert len(deprecation_warnings) == 1, (
                f"Expected 1 warning for SmartQADICycleResult, got {len(deprecation_warnings)}"
            )

    def test_answer_extractor_import_triggers_warning(self):
        """
        Test that importing from answer_extractor triggers a warning.
        """
        # Clear module cache
        if 'mad_spark_alt.core.answer_extractor' in sys.modules:
            del sys.modules['mad_spark_alt.core.answer_extractor']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Direct module import should trigger warning
            from mad_spark_alt.core import answer_extractor  # noqa: F401

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'answer_extractor' in str(warning.message)
            ]

            assert len(deprecation_warnings) == 1, (
                f"Expected 1 answer_extractor warning, got {len(deprecation_warnings)}"
            )

            warning_msg = str(deprecation_warnings[0].message)
            assert 'deprecated' in warning_msg.lower()
            assert 'v3.0.0' in warning_msg

    def test_robust_json_handler_import_triggers_warning(self):
        """
        Test that importing from robust_json_handler triggers a warning.
        """
        # Clear module cache
        if 'mad_spark_alt.core.robust_json_handler' in sys.modules:
            del sys.modules['mad_spark_alt.core.robust_json_handler']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from mad_spark_alt.core import robust_json_handler  # noqa: F401

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'robust_json_handler' in str(warning.message)
            ]

            assert len(deprecation_warnings) == 1, (
                f"Expected 1 robust_json_handler warning, got {len(deprecation_warnings)}"
            )

            warning_msg = str(deprecation_warnings[0].message)
            assert 'deprecated' in warning_msg.lower()
            assert 'v2.0.0' in warning_msg
            assert 'json_utils' in warning_msg

    def test_multiple_imports_one_warning_each(self):
        """
        Test that multiple imports of the same deprecated item show warning only once.
        """
        # Clear module cache
        for mod in list(sys.modules.keys()):
            if 'mad_spark_alt.core' in mod:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # First import
            from mad_spark_alt.core import SmartQADIOrchestrator  # noqa: F401, F811
            first_count = len([
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ])

            # Second import (same session) - should not add another warning
            # because module is already loaded
            from mad_spark_alt.core import SmartQADIOrchestrator  # noqa: F401, F811
            second_count = len([
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ])

            # Warning count should not increase on re-import
            assert first_count == second_count, (
                "Warnings should not repeat for already-imported modules"
            )

    def test_import_non_deprecated_no_warnings(self):
        """
        Test that importing non-deprecated core components doesn't trigger warnings.
        """
        # Clear module cache
        for mod in list(sys.modules.keys()):
            if 'mad_spark_alt.core' in mod:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import non-deprecated components
            from mad_spark_alt.core import UnifiedQADIOrchestrator  # noqa: F401
            from mad_spark_alt.core import SimpleQADIOrchestrator  # noqa: F401
            from mad_spark_alt.core import json_utils  # noqa: F401

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]

            assert len(deprecation_warnings) == 0, (
                f"Non-deprecated imports should not trigger warnings, got: "
                f"{[str(w.message) for w in deprecation_warnings]}"
            )

    def test_warning_stacklevel_points_to_user_code(self):
        """
        Test that warnings use correct stacklevel to point to user's import line.
        """
        # Clear module cache
        for mod in list(sys.modules.keys()):
            if 'mad_spark_alt.core' in mod:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This line should be identified as the source
            from mad_spark_alt.core import SmartQADIOrchestrator  # noqa: F401

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'Smart' in str(warning.message)
            ]

            if deprecation_warnings:
                # Warning should point to this file, not internal module files
                warning_file = deprecation_warnings[0].filename
                assert 'test_deprecation_warnings.py' in warning_file, (
                    f"Warning should point to user code (this test file), "
                    f"but points to: {warning_file}"
                )


class TestBackwardCompatibility:
    """Test that deprecated modules still work correctly."""

    def test_smart_orchestrator_still_functional(self):
        """
        Test that SmartQADIOrchestrator still works despite deprecation.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test

            from mad_spark_alt.core import SmartQADIOrchestrator

            # Should be able to instantiate
            orchestrator = SmartQADIOrchestrator()

            # Should have expected methods
            assert hasattr(orchestrator, 'run_qadi_cycle')
            assert callable(orchestrator.run_qadi_cycle)

    def test_answer_extractor_classes_still_accessible(self):
        """
        Test that answer_extractor classes are still accessible.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from mad_spark_alt.core.answer_extractor import (
                TemplateAnswerExtractor,
                EnhancedAnswerExtractor,
                ExtractedAnswer,
            )

            # Should be able to instantiate
            extractor = TemplateAnswerExtractor()
            assert hasattr(extractor, 'extract_answers')

            # Should be able to create data classes
            answer = ExtractedAnswer(
                content="test",
                confidence=0.8,
                source_phase="testing"
            )
            assert answer.content == "test"

    def test_robust_json_handler_still_functional(self):
        """
        Test that robust_json_handler functions still work.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from mad_spark_alt.core import robust_json_handler

            # Should have expected functions
            assert hasattr(robust_json_handler, 'extract_json_from_response')
            assert callable(robust_json_handler.extract_json_from_response)
            assert hasattr(robust_json_handler, 'safe_parse_ideas_array')
            assert callable(robust_json_handler.safe_parse_ideas_array)
