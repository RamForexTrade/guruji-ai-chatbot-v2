#!/usr/bin/env python3
"""
Enhanced setup script for JAI GURU DEV AI Chatbot
Handles dependency conflicts and provides multiple installation options.
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
from dotenv import load_dotenv

def install_requirements_with_fallback():
    """Install requirements with fallback options for dependency conflicts"""
    print("🔧 Installing required packages...")
    
    # Try method 1: Install with updated requirements
    try:
        print("📦 Attempting installation with specific versions...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Method 1 failed: {e}")
    
    # Try method 2: Install packages individually
    try:
        print("📦 Installing packages individually...")
        essential_packages = [
            "streamlit>=1.28.0",
            "langchain>=0.1.0", 
            "langchain-openai>=0.0.5",
            "langchain-groq>=0.0.1",
            "langchain-community>=0.0.10",
            "chromadb>=0.4.15",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0",
            "openai>=1.10.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0"
        ]
        
        for package in essential_packages:
            print(f"   Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    capture_output=True, text=True)
            except subprocess.CalledProcessError:
                print(f"   ⚠️ Failed to install {package}, continuing...")
        
        print("✅ Essential packages installed!")
        return True
        
    except Exception as e:
        print(f"❌ All installation methods failed: {e}")
        return False

def verify_installation():
    """Verify that core packages are installed"""
    print("🔍 Verifying installation...")
    
    required_packages = [
        'streamlit',
        'langchain', 
        'openai',
        'chromadb',
        'yaml',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All core packages verified!")
    return True

def verify_environment():
    """Verify environment variables"""
    print("🔍 Verifying environment setup...")
    
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    else:
        print("✅ OpenAI API key found")
    
    if not groq_key:
        print("⚠️  Warning: GROQ_API_KEY not found in environment")
    else:
        print("✅ Groq API key found")
    
    if not openai_key:
        print("❌ OpenAI API key is REQUIRED for embeddings!")
        return False
    
    return True

def verify_knowledge_base():
    """Verify knowledge base files"""
    print("📚 Verifying knowledge base...")
    
    kb_path = Path("Knowledge_Base")
    if not kb_path.exists():
        print(f"❌ Knowledge base directory not found: {kb_path}")
        return False
    
    md_files = list(kb_path.glob("*.md"))
    if not md_files:
        print("❌ No markdown files found in Knowledge_Base directory")
        return False
    
    print(f"✅ Found {len(md_files)} markdown files:")
    for file in md_files:
        print(f"   - {file.name}")
    
    return True

def test_config():
    """Test configuration file"""
    print("⚙️  Testing configuration...")
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['model_provider', 'rag', 'embeddings', 'ui', 'chatbot']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration file is valid")
        return True
        
    except FileNotFoundError:
        print("❌ config.yaml file not found")
        return False
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        return False

def main():
    """Main setup function"""
    print("🙏 JAI GURU DEV AI Chatbot - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Installation step
    print(f"\n📦 Installing Dependencies")
    if not install_requirements_with_fallback():
        print("❌ Failed to install dependencies")
        return False
    
    # Verification steps
    verification_steps = [
        ("Verify Installation", verify_installation),
        ("Verify Environment", verify_environment),
        ("Verify Knowledge Base", verify_knowledge_base),
        ("Test Configuration", test_config)
    ]
    
    all_passed = True
    for step_name, step_func in verification_steps:
        print(f"\n🔍 {step_name}...")
        if not step_func():
            all_passed = False
            print(f"❌ {step_name} failed!")
        else:
            print(f"✅ {step_name} passed!")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup completed successfully!")
        print("\n🚀 To start the chatbot:")
        print("   streamlit run chatbot.py")
        print("\n💡 Note: First run will build the vector database (~1-2 minutes)")
        print("\n🚀 For Railway deployment:")
        print("   1. Set OPENAI_API_KEY in Railway environment variables")
        print("   2. Optionally set GROQ_API_KEY")
        print("   3. Deploy!")
    else:
        print("❌ Setup encountered issues.")
        print("💡 Try running: streamlit run chatbot.py")
        print("💡 Check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    main()