#!/usr/bin/env python3
"""
Quick start script for MHPMR development setup
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and report status"""
    if description:
        print(f"\n{'='*60}")
        print(f"▶ {description}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Setup MHPMR development environment"""
    
    print("""
    ╔════════════════════════════════════════════════╗
    ║   MHPMR Development Setup Script              ║
    ║   Monocular 3D Human Pose and Mesh Recovery   ║
    ╚════════════════════════════════════════════════╝
    """)
    
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    print(f"Project root: {project_root}")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required!")
        return False
    
    # Create directories
    print("\n" + "="*60)
    print("▶ Creating project directories...")
    print("="*60)
    
    directories = [
        "src/mhpmr",
        "src/mhpmr/models",
        "src/mhpmr/datasets",
        "src/mhpmr/training",
        "src/mhpmr/visualization",
        "src/mhpmr/analysis",
        "tests",
        "notebooks",
        "data/models",
        "data/dataset",
        "data/pretrained",
        "data/examples",
        "tools",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Install dependencies
    should_install = input("\n▶ Install Python dependencies? (y/n): ").lower() == 'y'
    
    if should_install:
        if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                          "Upgrading pip..."):
            return False
        
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt",
                          "Installing dependencies..."):
            return False
        
        if not run_command(f"{sys.executable} -m pip install -e .",
                          "Installing MHPMR in development mode..."):
            return False
    
    # Verify installation
    print("\n" + "="*60)
    print("▶ Verifying installation...")
    print("="*60)
    
    verify_cmd = f"{sys.executable} -c 'import sys; sys.path.insert(0, \"src\"); import mhpmr; print(f\"✓ MHPMR v{{mhpmr.__version__}} loaded successfully\")'"
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Note: Full dependencies not installed yet")
        print(f"Install with: pip install -r requirements.txt")
    
    # Print next steps
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("""
Next steps:

1. Install dependencies:
   pip install -r requirements.txt

2. Install in development mode:
   pip install -e ".[dev]"

3. Run tests:
   pytest tests/ -v

4. Format code:
   black src/ tests/
   isort src/ tests/

5. Review documentation:
   - README.md        - User guide
   - IMPLEMENT.md     - Implementation specifications
   - DEVELOPMENT.md   - Development guide
   - RESEARCH_PAPER.md- Research details

6. Start implementing components following IMPLEMENT.md

Happy coding! 🚀
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
