#!/usr/bin/env python3
"""
Groq Model Troubleshooting Script
This script helps diagnose and fix issues with deprecated Groq models.
"""

import os
import yaml
from dotenv import load_dotenv
from langchain_groq import ChatGroq

def load_config():
    """Load the configuration file"""
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return None

def check_environment():
    """Check environment variables"""
    print("🔍 Checking environment variables...")
    load_dotenv()
    
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"   GROQ_API_KEY: {'✅ Found' if groq_key else '❌ Not found'}")
    print(f"   OPENAI_API_KEY: {'✅ Found' if openai_key else '❌ Not found'}")
    
    return groq_key, openai_key

def test_groq_models():
    """Test various Groq models to find working ones"""
    print("\n🧪 Testing Groq models...")
    
    # Load environment
    load_dotenv()
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not groq_key:
        print("❌ No Groq API key found. Please set GROQ_API_KEY in your .env file")
        return
    
    # List of models to test (current as of September 2025)
    test_models = [
        "llama-3.3-70b-versatile",      # Current recommended
        "llama-3.1-8b-instant",        # Current 8B model
        "llama-3.1-70b-versatile",     # Alternative 70B
        "mixtral-8x7b-32768",          # Mixtral model
        "gemma-7b-it",                 # Gemma model
    ]
    
    working_models = []
    
    for model in test_models:
        try:
            print(f"   Testing {model}...", end=" ")
            
            client = ChatGroq(
                model=model,
                max_tokens=10,
                temperature=0
            )
            
            # Test with a simple query
            response = client.invoke("Say hello")
            working_models.append(model)
            print("✅ Working")
            
        except Exception as e:
            error_msg = str(e)
            if "model_decommissioned" in error_msg or "400" in error_msg:
                print("❌ Deprecated/Decommissioned")
            else:
                print(f"❌ Error: {error_msg[:50]}...")
    
    print(f"\n✅ Working models found: {len(working_models)}")
    for model in working_models:
        print(f"   - {model}")
    
    return working_models

def check_deprecated_references():
    """Check for any hardcoded deprecated model references"""
    print("\n🔍 Checking for deprecated model references in code...")
    
    deprecated_models = [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "llama-3.1-405b-reasoning",
    ]
    
    files_to_check = [
        "config.yaml",
        "rag_system.py", 
        "chatbot.py",
        ".env"
    ]
    
    found_issues = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for deprecated_model in deprecated_models:
                    if deprecated_model in content:
                        found_issues.append(f"{file_path}: Found '{deprecated_model}'")
                        
            except Exception as e:
                print(f"   ⚠️ Could not read {file_path}: {e}")
    
    if found_issues:
        print("   ❌ Found deprecated model references:")
        for issue in found_issues:
            print(f"      {issue}")
    else:
        print("   ✅ No deprecated model references found")
    
    return found_issues

def suggest_fixes(working_models, found_issues):
    """Suggest fixes based on findings"""
    print("\n💡 Recommendations:")
    
    if found_issues:
        print("   1. Update deprecated model references:")
        for issue in found_issues:
            print(f"      - {issue}")
        if working_models:
            print(f"      → Replace with: {working_models[0]}")
    
    if working_models:
        print("   2. Update your config.yaml with a working model:")
        print("      model_provider:")
        print("        groq:")
        print(f"          model: {working_models[0]}")
        if len(working_models) > 1:
            print(f"          fallback_model: {working_models[1]}")
    
    print("   3. Clear any cached configurations:")
    print("      - Delete any .env files with hardcoded models")
    print("      - Restart your application")
    print("      - Check Railway/deployment environment variables")
    
    print("   4. If deployed on Railway:")
    print("      - Go to Railway dashboard → Variables")
    print("      - Remove any MODEL_NAME or similar variables")
    print("      - Redeploy the application")

def main():
    """Main troubleshooting function"""
    print("🔧 Groq Model Troubleshooting Script")
    print("=" * 50)
    
    # Check configuration
    config = load_config()
    if config:
        current_model = config.get('model_provider', {}).get('groq', {}).get('model', 'Not specified')
        print(f"📋 Current configured model: {current_model}")
    
    # Check environment
    groq_key, openai_key = check_environment()
    
    if not groq_key:
        print("\n❌ Cannot proceed without Groq API key")
        print("Please add GROQ_API_KEY to your .env file")
        return
    
    # Test models
    working_models = test_groq_models()
    
    # Check for deprecated references
    found_issues = check_deprecated_references()
    
    # Provide recommendations
    suggest_fixes(working_models, found_issues)
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    print(f"   Working models: {len(working_models) if working_models else 0}")
    print(f"   Issues found: {len(found_issues)}")
    
    if working_models and not found_issues:
        print("   ✅ Your setup should work correctly!")
    elif working_models:
        print("   ⚠️ Fix the issues above and try again")
    else:
        print("   ❌ No working models found - check your API key")

if __name__ == "__main__":
    main()
