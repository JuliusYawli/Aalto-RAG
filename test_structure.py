"""
Test script to validate RAG application structure
Tests basic imports and code structure without requiring API keys
"""
import os
import sys

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        '.env.example',
        '.gitignore',
        'src/__init__.py',
        'src/config.py',
        'src/document_loader.py',
        'src/vectorstore.py',
        'src/rag_chain.py',
        'rag_notebook.ipynb',
        'documents/ml_best_practices.txt',
        'documents/nlp_guide.txt',
        'documents/rag_systems.txt',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_syntax():
    """Test Python files for syntax errors"""
    import py_compile
    
    python_files = [
        'main.py',
        'src/__init__.py',
        'src/config.py',
        'src/document_loader.py',
        'src/vectorstore.py',
        'src/rag_chain.py',
    ]
    
    errors = []
    for file_path in python_files:
        try:
            py_compile.compile(file_path, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((file_path, str(e)))
    
    if errors:
        print("❌ Syntax errors found:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def test_requirements():
    """Test that requirements.txt has expected packages"""
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = [
        'langchain',
        'chromadb',
        'openai',
        'pypdf',
        'python-dotenv',
    ]
    
    missing = []
    for pkg in required_packages:
        if pkg not in content:
            missing.append(pkg)
    
    if missing:
        print("❌ Missing packages in requirements.txt:")
        for pkg in missing:
            print(f"  - {pkg}")
        return False
    else:
        print("✅ All required packages in requirements.txt")
        return True

def test_documentation():
    """Test that README has essential sections"""
    with open('README.md', 'r') as f:
        content = f.read()
    
    required_sections = [
        'Quick Start',
        'Installation',
        'Usage',
        'How It Works',
        'Configuration',
    ]
    
    missing = []
    for section in required_sections:
        if section not in content:
            missing.append(section)
    
    if missing:
        print("⚠️  Some documentation sections might be missing:")
        for section in missing:
            print(f"  - {section}")
        return True  # Not a critical failure
    else:
        print("✅ README contains all essential sections")
        return True

def test_sample_documents():
    """Test that sample documents exist and have content"""
    doc_files = [
        'documents/ml_best_practices.txt',
        'documents/nlp_guide.txt',
        'documents/rag_systems.txt',
    ]
    
    issues = []
    for file_path in doc_files:
        if not os.path.exists(file_path):
            issues.append(f"{file_path} does not exist")
        else:
            with open(file_path, 'r') as f:
                content = f.read()
            if len(content) < 100:
                issues.append(f"{file_path} has less than 100 characters")
    
    if issues:
        print("❌ Issues with sample documents:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All sample documents exist and have content")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG Application Structure Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_syntax),
        ("Requirements", test_requirements),
        ("Documentation", test_documentation),
        ("Sample Documents", test_sample_documents),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting: {name}")
        print("-" * 60)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("The RAG application structure is properly set up.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up OpenAI API key in .env file")
        print("3. Index documents: python main.py --index")
        print("4. Start asking questions!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
