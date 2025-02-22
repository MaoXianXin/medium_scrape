import fitz  # PyMuPDF
import tiktoken
from pathlib import Path

def process_pdf(pdf_path, output_dir=None):
    """
    Read a PDF file, extract and merge all text, then segment it based on token count
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the segmented text files
    
    Returns:
        tuple: List of text segments, total tokens, and the output directory
    """
    # Initialize tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")
    max_tokens_per_segment = 7000
    
    # Setup output directory
    pdf_path = Path(pdf_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_segments"
    output_dir.mkdir(exist_ok=True)
    
    # Check if file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Extract text from PDF
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")
    
    # Calculate total tokens
    total_tokens = len(tokenizer.encode(text, disallowed_special=()))
    
    # Segment the text
    tokens = tokenizer.encode(text, disallowed_special=())
    segments = []
    current_segment_tokens = []
    current_length = 0
    
    for token in tokens:
        if current_length >= max_tokens_per_segment:
            segment_text = tokenizer.decode(current_segment_tokens)
            segments.append(segment_text)
            current_segment_tokens = [token]
            current_length = 1
        else:
            current_segment_tokens.append(token)
            current_length += 1
    
    # Handle remaining tokens
    if current_segment_tokens:
        if len(current_segment_tokens) >= 1000:
            segment_text = tokenizer.decode(current_segment_tokens)
            segments.append(segment_text)
        elif segments:  # If there's a previous segment, merge with it
            last_segment = segments[-1]
            additional_text = tokenizer.decode(current_segment_tokens)
            segments[-1] = last_segment + additional_text
    
    # Save segments to files
    for i, segment in enumerate(segments, 1):
        output_file = output_dir / f"segment_{i:03d}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(segment)
    
    return segments, total_tokens

# Example usage
if __name__ == "__main__":
    try:
        pdf_path = input("Please enter the path to your PDF file: ").strip()
        output_dir = input("Please enter the output directory (press Enter for default): ").strip()
        output_dir = output_dir if output_dir else None
        
        segments, total_tokens = process_pdf(pdf_path, output_dir)
        
        print(f"成功将PDF分割为 {len(segments)} 个部分")
        print(f"分段文件保存在: {output_dir}")
        for i, segment in enumerate(segments, 1):
            print(f"\n分段 {i} 长度 (tokens): {len(tiktoken.encoding_for_model('gpt-4-turbo').encode(segment))}")
            print(f"前100个字符: {segment[:100]}...")
        
        print(f"\n总tokens数量: {total_tokens}")
    except Exception as e:
        print(f"错误: {str(e)}")