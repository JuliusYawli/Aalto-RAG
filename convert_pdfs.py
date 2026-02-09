#!/usr/bin/env python3
"""
Convert all PDFs in documents folder to text files
"""
import os
from pathlib import Path
from pypdf import PdfReader

documents_path = "/Users/jy/Documents/GitHub/Aalto-RAG/documents"

# Get all PDF files
pdf_files = list(Path(documents_path).glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")

for pdf_file in pdf_files:
    print(f"\nConverting: {pdf_file.name}")
    
    try:
        # Read PDF
        reader = PdfReader(str(pdf_file))
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        
        # Write to text file
        txt_file = pdf_file.parent / f"{pdf_file.stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✅ Converted to: {txt_file.name}")
        
    except Exception as e:
        print(f"❌ Error converting {pdf_file.name}: {str(e)}")

print("\n✅ All PDFs converted to text files!")
