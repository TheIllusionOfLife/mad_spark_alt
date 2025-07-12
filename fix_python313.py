#!/usr/bin/env python3
"""
Fix Python 3.13 compatibility issues by finding and fixing all problematic exception patterns.
"""

import os
import re
from pathlib import Path

def fix_exception_patterns(file_path):
    """Fix problematic exception patterns in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Fix compound exception catching with Exception
    # except (SomeError, Exception) -> except Exception
    content = re.sub(
        r'except\s*\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\s*,\s*)*Exception\s*\)\s*as\s+(\w+):',
        r'except Exception as \2:',
        content
    )
    
    # Pattern 2: Fix any remaining compound exceptions that might cause issues
    # This is more conservative - only fix if we see specific problematic patterns
    content = re.sub(
        r'except\s*\(\s*json\.JSONDecodeError\s*,\s*Exception\s*\)\s*as\s+(\w+):',
        r'except Exception as \1:',
        content
    )
    
    # Pattern 3: Fix type: ignore on except lines (can cause issues in Python 3.13)
    content = re.sub(
        r'except\s+([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*#\s*type:\s*ignore',
        r'except \1:',
        content
    )
    
    if content != original_content:
        print(f"Fixed: {file_path}")
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Find and fix all Python files with problematic exception patterns."""
    src_dir = Path(__file__).parent / "src"
    
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        if fix_exception_patterns(py_file):
            fixed_files.append(py_file)
    
    print(f"\nFixed {len(fixed_files)} files")
    for f in fixed_files:
        print(f"  - {f.relative_to(src_dir.parent)}")

if __name__ == "__main__":
    main()