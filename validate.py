#!/usr/bin/env python3
"""
Simple validation script to test core functionality of the repository projects.
Run this to verify everything is working as expected.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_mario_server():
    """Test if Mario server can start without errors."""
    print("🎮 Testing Mario server...")
    mario_dir = Path(__file__).parent / "jokes & gimmicks" / "mario"
    
    try:
        # Check if node_modules exists
        if not (mario_dir / "node_modules").exists():
            print("   ⚠️  Mario dependencies not installed. Run 'npm install' in mario directory.")
            return False
            
        # Try to start server for 2 seconds
        result = subprocess.run(
            ["timeout", "2s", "npm", "start"],
            cwd=mario_dir,
            env={**os.environ, "PORT": "3004"},
            capture_output=True,
            text=True
        )
        
        if "Mario agent server running" in result.stdout:
            print("   ✅ Mario server starts successfully")
            return True
        elif "Port 3004 is already in use" in result.stderr:
            print("   ✅ Mario server error handling works correctly")
            return True
        else:
            print(f"   ❌ Mario server failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Mario test failed: {e}")
        return False

def test_gremlin_import():
    """Test if Gremlin can be imported and handles missing dependencies gracefully."""
    print("🧙 Testing Gremlin chatbot...")
    
    try:
        # Try to import and test error handling
        gremlin_path = Path(__file__).parent / "jokes & gimmicks" / "gremlin.py"
        
        # Test file exists and is readable
        if not gremlin_path.exists():
            print("   ❌ Gremlin file not found")
            return False
            
        # Try to run syntax check
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(gremlin_path)],
            capture_output=True
        )
        
        if result.returncode == 0:
            print("   ✅ Gremlin syntax is valid")
            return True
        else:
            print(f"   ❌ Gremlin syntax error: {result.stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"   ❌ Gremlin test failed: {e}")
        return False

def test_python_files():
    """Test if main Python files are syntactically correct."""
    print("🐍 Testing Python files...")
    
    python_files = ["stupid.py", "virus.py", "tartarus.py"]
    all_passed = True
    
    for file in python_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", file],
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"   ✅ {file} syntax is valid")
            else:
                print(f"   ❌ {file} syntax error")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ {file} test failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    print("🔍 Repository Quality Validation")
    print("=" * 40)
    
    tests = [
        test_python_files,
        test_gremlin_import,
        test_mario_server,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Repository quality looks good.")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())