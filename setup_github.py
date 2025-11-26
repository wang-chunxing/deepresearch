#!/usr/bin/env python3
"""
Setup script to initialize GitHub repository for the Deep Research Report Generation Agent System
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture_output=True, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def main():
    print("Setting up GitHub repository for Deep Research Report Generation Agent System...")
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("Error: This script should be run from the project root directory")
        sys.exit(1)
    
    # Initialize git repository if not already initialized
    if not Path(".git").exists():
        print("Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command("git commit -m \"Initial commit: Deep Research Report Generation Agent System\"")
    else:
        print("Git repository already exists, adding and committing changes...")
        run_command("git add .")
        run_command("git commit -m \"Add implementation files for Deep Research Report Generation Agent System\" || true")
    
    print("\n" + "="*60)
    print("GITHUB REPOSITORY SETUP INSTRUCTIONS")
    print("="*60)
    print("\nTo complete the GitHub setup, please follow these steps manually:")
    print()
    print("1. Go to https://github.com and create a new repository")
    print("2. Name it 'deep-research-agent' (or your preferred name)")
    print("3. Do NOT initialize with README, .gitignore, or license")
    print()
    print("Then run these commands in your terminal:")
    print()
    print("   # Add the remote origin")
    print("   git remote add origin https://github.com/YOUR_USERNAME/deep-research-agent.git")
    print()
    print("   # Push the code to GitHub")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("Alternatively, if using SSH:")
    print("   git remote add origin git@github.com:YOUR_USERNAME/deep-research-agent.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("="*60)
    print("PROJECT STRUCTURE COMPLETED")
    print("="*60)
    print("\nThe following components have been created:")
    print("- Core architecture (API, Agents, Memory, Generation, Tools)")
    print("- Configuration system")
    print("- Documentation (README, design docs)")
    print("- Environment configuration")
    print("- Requirements file with all dependencies")
    print("\nThis implementation follows the architecture design from the research report")
    print("and includes all modules for Phase 1 of the implementation roadmap.")
    print()


if __name__ == "__main__":
    main()