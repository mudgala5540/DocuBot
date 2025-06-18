#!/usr/bin/env python3
"""
Setup script for PDF Intelligence Agent
"""

import os
import sys
import subprocess
import platform

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    print("Installing system dependencies...")
    
    if system == "linux":
        # Ubuntu/Debian
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr-eng"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libgl1-mesa-glx"], check=True)  # For OpenCV
            print("✅ System dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install system dependencies. Please install manually:")
            print("sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx")
    
    elif system == "darwin":  # macOS
        try:
            subprocess.run(["brew", "install", "tesseract"], check=True)
            print("✅ Tesseract installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Tesseract. Please install manually:")
            print("brew install tesseract")
    
    elif system == "windows":
        print("⚠️  For Windows, please install Tesseract manually:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install to C:\\Program Files\\Tesseract-OCR\\")
        print("3. Add to system PATH or update TESSERACT_PATH in .env file")

def setup_environment():
    """Setup Python environment and install packages"""
    print("Setting up Python environment...")
    
    # Upgrade pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    print("✅ Python packages installed successfully!")

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Google API Key for Gemini\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n\n")
            f.write("# Optional: Tesseract path (if not in system PATH)\n")
            f.write("# TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n")
        
        print("✅ .env file created. Please add your GOOGLE_API_KEY!")
    else:
        print("✅ .env file already exists.")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "uploads",
        "cache",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project directories created!")

def test_installation():
    """Test if everything is installed correctly"""
    print("Testing installation...")
    
    try:
        # Test imports
        import streamlit
        import fitz  # PyMuPDF
        import PIL
        import pytesseract
        import cv2
        import sentence_transformers
        import faiss
        import google.generativeai
        
        print("✅ All Python packages imported successfully!")
        
        # Test Tesseract
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR is working!")
        except Exception as e:
            print(f"❌ Tesseract OCR test failed: {e}")
            print("Please ensure Tesseract is installed and in PATH")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 PDF Intelligence Agent Setup")
    print("=" * 40)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Setup Python environment
    setup_environment()
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your GOOGLE_API_KEY to the .env file")
        print("2. Run: streamlit run main.py")
    else:
        print("\n❌ Setup completed with errors. Please check the issues above.")

if __name__ == "__main__":
    main()