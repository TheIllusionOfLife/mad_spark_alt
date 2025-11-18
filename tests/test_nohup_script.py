"""
Tests for run_nohup.sh script functionality
"""

import os
import subprocess
from pathlib import Path
import pytest
import re
import time


class TestNohupScript:
    """Test the nohup script functionality."""
    
    @pytest.fixture
    def script_dir(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def outputs_dir(self, script_dir):
        """Ensure outputs directory exists."""
        outputs = script_dir / "outputs"
        outputs.mkdir(exist_ok=True)
        return outputs
    
    def test_script_exists_after_rename(self, script_dir):
        """Test that run_nohup.sh exists after renaming."""
        script_path = script_dir / "scripts" / "run_nohup.sh"
        assert script_path.exists(), "scripts/run_nohup.sh should exist"
        assert os.access(script_path, os.X_OK), "run_nohup.sh should be executable"
    
    def test_outputs_directory_used(self, script_dir, outputs_dir):
        """Test that script saves output to outputs directory."""
        # This test will verify the script creates files in outputs/
        # We'll check this by examining the script content
        script_path = script_dir / "scripts" / "run_nohup.sh"
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
                assert 'outputs/' in content, "Script should use outputs directory"
    
    def test_script_handles_no_arguments(self, script_dir):
        """Test script shows usage when no arguments provided."""
        script_path = script_dir / "scripts" / "run_nohup.sh"
        if not script_path.exists():
            pytest.skip("Script not yet created")
            
        result = subprocess.run(
            [str(script_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Should exit with error code 1"
        assert "Usage:" in result.stdout, "Should show usage message"
        assert "Examples:" in result.stdout, "Should show examples"
    
    def test_script_checks_venv(self, script_dir, tmp_path):
        """Test script checks for virtual environment."""
        # Create a temporary script location without .venv
        temp_script = tmp_path / "run_nohup.sh"

        script_path = script_dir / "scripts" / "run_nohup.sh"
        if not script_path.exists():
            pytest.skip("Script not yet created")
            
        # Copy script to temp location
        temp_script.write_text(script_path.read_text())
        temp_script.chmod(0o755)
        
        # Run from temp directory (no .venv)
        result = subprocess.run(
            [str(temp_script), "test"],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Should exit with error"
        assert "Virtual environment not found" in result.stdout, "Should warn about missing venv"
    
    def test_output_file_naming(self, script_dir, outputs_dir):
        """Test that output files have timestamp in name."""
        # Check script content for timestamp pattern
        script_path = script_dir / "scripts" / "run_nohup.sh"
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
                assert 'date +%Y%m%d_%H%M%S' in content, "Should use timestamp in filename"
                assert 'OUTPUT_FILE=' in content, "Should define OUTPUT_FILE variable"
    
    def test_script_shows_monitoring_instructions(self, script_dir):
        """Test that script provides monitoring instructions."""
        script_path = script_dir / "scripts" / "run_nohup.sh"
        if not script_path.exists():
            pytest.skip("Script not yet created")
            
        # We'll verify by checking script content
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'cat' in content, "Should show cat command"
            assert 'ps -p' in content, "Should show ps command"


class TestEvolutionScript:
    """Test the direct evolution script."""
    
    @pytest.fixture
    def script_dir(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    def test_script_has_timeout_warning(self, script_dir):
        """Test that run_evolution.sh warns about potential timeout."""
        script_path = script_dir / "scripts" / "run_evolution.sh"
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
                # Should have warning about 2-minute timeout
                assert 'timeout' in content.lower() or '2 minute' in content, \
                    "Script should warn about potential timeout"


class TestDocumentationUpdates:
    """Test that documentation is properly updated."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    def test_evolution_timeout_fix_updated(self, project_root):
        """Test EVOLUTION_TIMEOUT_FIX.md has correct information."""
        doc_path = project_root / "EVOLUTION_TIMEOUT_FIX.md"
        if doc_path.exists():
            content = doc_path.read_text()
            
            # Should mention that direct execution doesn't work
            assert "direct python execution" in content.lower() or "still times out" in content.lower(), \
                "Should mention direct execution doesn't work"
            
            # Should emphasize nohup as the working solution
            assert "nohup" in content and ("only" in content.lower() or "working solution" in content.lower()), \
                "Should emphasize nohup as the working solution"
            
            # Should reference run_nohup.sh (not run_evolution_nohup.sh)
            assert "run_nohup.sh" in content, "Should reference renamed script"
            assert "run_evolution_nohup.sh" not in content, "Should not reference old script name"
    
    def test_readme_has_known_issues(self, project_root):
        """Test README.md has Known Issues section."""
        readme_path = project_root / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            
            # Should have Known Issues section
            assert "known issues" in content.lower() or "timeout" in content.lower(), \
                "README should mention timeout issue"
            
            # Should reference the nohup workaround
            if "timeout" in content.lower():
                assert "nohup" in content.lower() or "run_nohup" in content.lower(), \
                    "README should reference nohup workaround"
    
    def test_gitignore_includes_outputs(self, project_root):
        """Test .gitignore includes outputs directory."""
        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            # Should ignore outputs directory
            assert "outputs/" in content or "outputs" in content, \
                ".gitignore should include outputs directory"


class TestOutputDirectory:
    """Test outputs directory handling."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    def test_outputs_directory_exists(self, project_root):
        """Test that outputs directory exists."""
        outputs_dir = project_root / "outputs"
        assert outputs_dir.exists(), "outputs directory should exist"
        assert outputs_dir.is_dir(), "outputs should be a directory"
    
    def test_old_output_file_moved(self, project_root):
        """Test that old output file is moved to outputs directory."""
        old_file = project_root / "evolution_output_20250731_185012.txt"
        new_file = project_root / "outputs" / "evolution_output_20250731_185012.txt"
        
        # Either the old file doesn't exist (moved) or new file exists (moved to)
        assert not old_file.exists() or new_file.exists(), \
            "Old output file should be moved to outputs directory"


@pytest.mark.integration
class TestScriptIntegration:
    """Integration tests for the scripts with real commands."""
    
    @pytest.fixture
    def script_dir(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    def test_nohup_script_creates_output_file(self, script_dir):
        """Test that nohup script creates an output file and cleans it up."""
        script_path = script_dir / "scripts" / "run_nohup.sh"
        if not script_path.exists():
            pytest.skip("Script not yet created")

        # Run a simple command that should complete quickly
        result = subprocess.run(
            [str(script_path), "test prompt", "--help"],
            capture_output=True,
            text=True,
            check=True
        )

        # 1. Parse PID and output file path from stdout
        pid_match = re.search(r"Process started with PID: (\d+)", result.stdout)
        assert pid_match, "Should show process ID"
        pid = int(pid_match.group(1))

        output_file_match = re.search(r"Output will be saved to: (outputs/.+\.txt)", result.stdout)
        assert output_file_match, "Should indicate output location"
        output_file_path = script_dir / output_file_match.group(1)

        try:
            # 2. Wait for the background process to finish (poll for up to 10s)
            for _ in range(100):
                # Use ps to check if process exists. Exit code 0 means it exists.
                proc_check = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
                if proc_check.returncode != 0:
                    break  # Process finished
                time.sleep(0.1)
            else:
                pytest.fail(f"Process {pid} did not finish in time.")

            # 3. Assert that the output file exists
            assert output_file_path.exists(), f"Output file {output_file_path} was not created"

            # Optional: check content
            content = output_file_path.read_text()
            # After CLI consolidation, command is now 'msa'
            assert "usage: msa" in content.lower()

        finally:
            # 4. Clean up the created output file
            if output_file_path.exists():
                output_file_path.unlink()