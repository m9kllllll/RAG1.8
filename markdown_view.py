import os
from docling.document_converter import DocumentConverter

def review_docling_conversion(file_path: str, output_md_path: str = "review_output.md") -> str:
    """
    Converts a document (PDF, DOCX, image, etc.) using Docling, 
    exports the structured content to Markdown, and saves it locally for review.
    """
    if not os.path.exists(file_path) and not file_path.startswith("http"):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Initialize the Docling document converter
    converter = DocumentConverter()
    
    print(f"Converting '{file_path}' using Docling...")
    result = converter.convert(file_path)
    
    # Export parsed content to Markdown format
    markdown_content = result.document.export_to_markdown()
    
    # Save to file for visual inspection in any Markdown viewer
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Markdown output saved successfully to: {output_md_path}")
    return output_md_path

# Example Usage:
# review_docling_conversion("path/to/document.pdf", "preview_output.md")