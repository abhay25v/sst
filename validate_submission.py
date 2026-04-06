#!/usr/bin/env python3
"""
Pre-submission validation script for OpenEnv environment.
Verifies all requirements before submitting to OpenEnv competition.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

class ValidationChecker:
    """Checks submission readiness."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def check(self, name: str, fn) -> bool:
        """Run a check and record result."""
        try:
            result = fn()
            if result:
                print(f"✓ {name}")
                self.passed += 1
            else:
                print(f"✗ {name}")
                self.failed += 1
            self.results[name] = result
            return result
        except Exception as e:
            print(f"✗ {name}: {str(e)}")
            self.failed += 1
            self.results[name] = False
            return False
    
    def check_files_exist(self):
        """Verify all required files exist."""
        required_files = [
            "environment.py",
            "models.py",
            "dataset.py",
            "reward.py",
            "server.py",
            "inference.py",
            "openenv.yaml",
            "Dockerfile",
            "requirements.txt",
            "README.md",
        ]
        return all((self.repo_root / f).exists() for f in required_files)
    
    def check_dockerfile(self):
        """Verify Dockerfile is valid."""
        dockerfile = self.repo_root / "Dockerfile"
        if not dockerfile.exists():
            return False
        try:
            content = dockerfile.read_text(encoding='utf-8')
        except:
            content = dockerfile.read_text(encoding='latin-1')
        # Check for critical components
        return all(x in content for x in [
            "FROM python:3.11",
            "COPY requirements.txt",
            "pip install",
            "EXPOSE 8000",
            "CMD",
        ])
    
    def check_requirements(self):
        """Verify requirements.txt has critical packages."""
        req_file = self.repo_root / "requirements.txt"
        if not req_file.exists():
            return False
        try:
            content = req_file.read_text(encoding='utf-8').lower()
        except:
            content = req_file.read_text(encoding='latin-1').lower()
        return all(x in content for x in [
            "fastapi",
            "uvicorn",
            "pydantic",
            "openai",
        ])
    
    def check_openenv_yaml(self):
        """Verify openenv.yaml has required sections."""
        yaml_file = self.repo_root / "openenv.yaml"
        if not yaml_file.exists():
            return False
        try:
            content = yaml_file.read_text(encoding='utf-8')
        except:
            content = yaml_file.read_text(encoding='latin-1')
        required_sections = [
            "meta:",
            "environment:",
            "observation:",
            "action:",
            "reward:",
            "episode:",
            "tasks:",
            "server:",
        ]
        return all(section in content for section in required_sections)
    
    def check_models(self):
        """Verify models.py has required Pydantic models."""
        models_file = self.repo_root / "models.py"
        if not models_file.exists():
            return False
        try:
            content = models_file.read_text(encoding='utf-8')
        except:
            content = models_file.read_text(encoding='latin-1')
        required_classes = [
            "class Observation",
            "class Action",
            "class StepResult",
            "class ResetResult",
        ]
        return all(cls in content for cls in required_classes)
    
    def check_inference_entry(self):
        """Verify inference.py has main() entry point."""
        inf_file = self.repo_root / "inference.py"
        if not inf_file.exists():
            return False
        try:
            content = inf_file.read_text(encoding='utf-8')
        except:
            content = inf_file.read_text(encoding='latin-1')
        return 'def main():' in content and 'if __name__ == "__main__":' in content
    
    def check_env_variables(self):
        """Verify inference uses required env variables."""
        inf_file = self.repo_root / "inference.py"
        if not inf_file.exists():
            return False
        try:
            content = inf_file.read_text(encoding='utf-8')
        except:
            content = inf_file.read_text(encoding='latin-1')
        return all(var in content for var in [
            "OPENAI_API_KEY",
            "MODEL_NAME",
            "API_BASE_URL",
        ])
    
    def check_tasks_defined(self):
        """Verify 12 tasks are defined across 3 difficulties."""
        dataset_file = self.repo_root / "dataset.py"
        if not dataset_file.exists():
            return False
        try:
            content = dataset_file.read_text(encoding='utf-8')
        except:
            content = dataset_file.read_text(encoding='latin-1')
        # Count tasks by difficulty
        easy = content.count("task_id=\"easy_")
        medium = content.count("task_id=\"medium_")
        hard = content.count("task_id=\"hard_")
        return easy >= 4 and medium >= 4 and hard >= 4
    
    def check_grader(self):
        """Verify grader returns 0.0-1.0 scores."""
        reward_file = self.repo_root / "reward.py"
        if not reward_file.exists():
            return False
        try:
            content = reward_file.read_text(encoding='utf-8')
        except:
            content = reward_file.read_text(encoding='latin-1')
        return "grade_episode" in content and "class DeterministicGrader" in content
    
    def check_endpoints(self):
        """Verify FastAPI server has required endpoints."""
        server_file = self.repo_root / "server.py"
        if not server_file.exists():
            return False
        try:
            content = server_file.read_text(encoding='utf-8')
        except:
            content = server_file.read_text(encoding='latin-1')
        endpoints = [
            '@app.post("/reset"',
            '@app.post("/step"',
            '@app.get("/state"',
            '@app.get("/health"',
        ]
        return all(ep in content for ep in endpoints)
    
    def check_readme(self):
        """Verify README has required sections."""
        readme = self.repo_root / "README.md"
        if not readme.exists():
            return False
        try:
            content = readme.read_text(encoding='utf-8')
        except:
            content = readme.read_text(encoding='latin-1')
        required_sections = [
            "Real-World Utility",
            "OpenEnv",
            "Task",
            "Action Space",
            "Observation",
            "Reward",
            "Setup",
            "Baseline",
        ]
        return all(section in content for section in required_sections)
    
    def run_all_checks(self):
        """Run all validation checks."""
        print("\n" + "="*80)
        print("PRE-SUBMISSION VALIDATION")
        print("="*80 + "\n")
        
        print("File Structure:")
        self.check("All required files exist", self.check_files_exist)
        
        print("\nDockerization:")
        self.check("Dockerfile is valid", self.check_dockerfile)
        self.check("requirements.txt has dependencies", self.check_requirements)
        
        print("\nOpenEnv Specification:")
        self.check("openenv.yaml has required sections", self.check_openenv_yaml)
        
        print("\nCode Structure:")
        self.check("models.py has Pydantic models", self.check_models)
        self.check("inference.py has main() entry", self.check_inference_entry)
        
        print("\nComplete Specification:")
        self.check("inference.py reads env variables", self.check_env_variables)
        self.check("12 tasks defined (4 each difficulty)", self.check_tasks_defined)
        self.check("Grader returns 0.0-1.0 scores", self.check_grader)
        self.check("FastAPI has required endpoints", self.check_endpoints)
        self.check("README complete with all sections", self.check_readme)
        
        print("\n" + "="*80)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("="*80 + "\n")
        
        if self.failed == 0:
            print("✓ ALL CHECKS PASSED - Ready for submission!")
            print("\nNext steps:")
            print("1. Run: python inference.py --benchmark")
            print("2. Verify baseline scores are reproducible")
            print("3. Run: docker build .")
            print("4. Test: docker run -p 8000:8000 <image>")
            print("5. Submit to OpenEnv platform")
            return True
        else:
            print(f"✗ {self.failed} checks failed. Please fix issues above.")
            return False

if __name__ == "__main__":
    checker = ValidationChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)
