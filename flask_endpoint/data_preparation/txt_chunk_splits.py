import logging
import os
from transformers import AutoTokenizer  # Import the tokenizer
import argparse # Import argparse for command-line arguments

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Tokenizer Setup ---
# Load the tokenizer *once*
TOKENIZER_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Or choose another suitable model
MAX_CHUNK_TOKENS = 200 # Default max tokens per chunk
try:
    logger.info(f"Loading tokenizer: {TOKENIZER_MODEL_NAME}...")
    # Only the tokenizer is needed for counting tokens
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
    # Exit or handle error appropriately if tokenizer is essential
    tokenizer = None # Set tokenizer to None if loading fails
    logger.warning("Tokenizer could not be loaded. Chunking will not work.")
    # Depending on the use case, you might want to exit(1) here
# --- End Tokenizer Setup ---


def chunk_text_by_sentence(text: str, tokenizer: AutoTokenizer, max_tokens: int = 200) -> list[str]:
    """
    Splits text by sentences (using '.') and groups them into chunks
    respecting the max_tokens limit. Sentences exceeding max_tokens
    individually are discarded.

    Args:
        text: The input text string.
        tokenizer: The Hugging Face tokenizer instance.
        max_tokens: The maximum number of tokens allowed per chunk.

    Returns:
        A list of text chunks.
    """
    if not text or not tokenizer: # Check if tokenizer is available
        logger.warning("Input text is empty or tokenizer is not loaded. Returning empty list.")
        return []

    # Split by period and keep the period. Filter out empty strings.
    sentences = [s.strip() + "." for s in text.split('.') if s.strip()]
    # Handle case where the original text might not end with a period
    if text.strip() and not text.strip().endswith('.'):
         # Get the last part if it wasn't captured by split('.')
         last_part = text.split('.')[-1].strip()
         if last_part:
             # Find the last sentence that ends with a period
             last_sentence_with_period = ""
             if sentences:
                 last_sentence_with_period = sentences[-1]

             # Check if the last part is already the content of the last sentence
             if not last_sentence_with_period.startswith(last_part):
                 sentences.append(last_part) # Add the part without a period if it's new content


    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    logger.info(f"Starting chunking process with max_tokens={max_tokens}.")
    for i, sentence in enumerate(sentences):
        try:
            # Calculate tokens for the current sentence
            # Use add_special_tokens=False for more accurate counting within context
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

            if sentence_tokens > max_tokens:
                logger.warning(f"Sentence {i+1} exceeds token limit ({sentence_tokens} > {max_tokens}). Discarding: '{sentence[:80]}...'")
                continue # Skip this sentence

            # Check if adding the sentence exceeds the limit for the current chunk
            if current_chunk_tokens + sentence_tokens <= max_tokens:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
            else:
                # Current chunk is full, finalize it
                if current_chunk_sentences:
                    chunk = " ".join(current_chunk_sentences)
                    chunks.append(chunk)
                    logger.debug(f"Created chunk with {current_chunk_tokens} tokens: '{chunk[:80]}...'")

                # Start a new chunk with the current sentence
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
                # Check if the new sentence *itself* exceeds the limit (redundant due to check above, but safe)
                if current_chunk_tokens > max_tokens:
                     logger.warning(f"Sentence {i+1} starting new chunk exceeds token limit ({current_chunk_tokens} > {max_tokens}). Discarding: '{sentence[:80]}...'")
                     current_chunk_sentences = []
                     current_chunk_tokens = 0


        except Exception as e:
            logger.error(f"Error processing sentence {i+1}: '{sentence[:80]}...'. Error: {e}", exc_info=True)
            # Decide how to handle errors, e.g., skip the sentence
            continue

    # Add the last remaining chunk if any
    if current_chunk_sentences:
        chunk = " ".join(current_chunk_sentences)
        chunks.append(chunk)
        logger.debug(f"Created final chunk with {current_chunk_tokens} tokens: '{chunk[:80]}...'")

    logger.info(f"Chunking complete. Generated {len(chunks)} chunks.")
    return chunks


"""
python txt_chunk_splits.py --input-dir /home/mao/datasets/machine_translation/articles --output-dir /home/mao/datasets/machine_translation/en_txt_chunks --max-tokens 200
"""

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Chunk text files by sentence based on token limits.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input .txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output chunked files.")
    parser.add_argument("--max-tokens", type=int, default=MAX_CHUNK_TOKENS,
                        help=f"Maximum tokens per chunk (default: {MAX_CHUNK_TOKENS}).")
    args = parser.parse_args()

    # --- Input/Output Validation and Setup ---
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        exit(1) # Exit if input directory doesn't exist

    if not tokenizer:
        logger.error("Tokenizer failed to load. Cannot proceed with chunking.")
        exit(1) # Exit if tokenizer isn't available

    try:
        os.makedirs(args.output_dir, exist_ok=True) # Create output directory if it doesn't exist
        logger.info(f"Output directory created or already exists: {args.output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}", exc_info=True)
        exit(1) # Exit if output directory cannot be created

    logger.info(f"Starting processing of files in: {args.input_dir}")
    logger.info(f"Output will be saved to: {args.output_dir}")
    logger.info(f"Using max_tokens per chunk: {args.max_tokens}")

    # --- File Processing Loop ---
    processed_files_count = 0
    total_chunks_saved = 0 # Keep track of total chunks saved across all files
    for filename in os.listdir(args.input_dir):
        # Process only files ending with .txt (case-insensitive)
        if filename.lower().endswith(".txt"):
            input_filepath = os.path.join(args.input_dir, filename)
            base_filename = os.path.splitext(filename)[0] # Get filename without extension

            logger.info(f"Processing file: {input_filepath}")

            try:
                # Read the content of the input file
                with open(input_filepath, 'r', encoding='utf-8') as f_in:
                    text_content = f_in.read()

                # Skip empty files
                if not text_content.strip():
                    logger.warning(f"File is empty, skipping: {input_filepath}")
                    continue

                # Perform the chunking using the existing function
                text_chunks = chunk_text_by_sentence(text_content, tokenizer, args.max_tokens)

                # Skip if no chunks were generated
                if not text_chunks:
                    logger.warning(f"No chunks generated for file: {input_filepath}")
                    continue

                # --- Write each chunk to a separate file ---
                chunks_saved_for_file = 0
                for i, chunk in enumerate(text_chunks):
                    # Construct output filename for each chunk
                    output_filename = f"{base_filename}_chunk_{i+1}.txt"
                    output_filepath = os.path.join(args.output_dir, output_filename)

                    try:
                        # Write the current chunk to its own file
                        with open(output_filepath, 'w', encoding='utf-8') as f_out:
                            f_out.write(chunk)
                        logger.debug(f"Saved chunk {i+1}/{len(text_chunks)} to: {output_filepath}")
                        chunks_saved_for_file += 1
                    except IOError as e:
                        logger.error(f"Error writing chunk {i+1} for {filename} to {output_filepath}: {e}", exc_info=True)
                        # Decide if you want to stop processing this file or continue with other chunks
                        # For now, we'll log the error and continue

                logger.info(f"Successfully saved {chunks_saved_for_file}/{len(text_chunks)} chunks for file: {filename}")
                if chunks_saved_for_file > 0:
                    processed_files_count += 1
                    total_chunks_saved += chunks_saved_for_file
                # --- End writing chunks ---

            except FileNotFoundError:
                # This might happen if the file is deleted between listdir and open
                logger.error(f"File not found during processing: {input_filepath}")
            except IOError as e:
                # Error reading the input file
                logger.error(f"Error reading file {input_filepath}: {e}", exc_info=True)
            except Exception as e:
                # Catch any other unexpected errors during processing of a single file
                logger.error(f"An unexpected error occurred processing {input_filepath}: {e}", exc_info=True)
                # Continue to the next file

    logger.info(f"Processing complete. Processed {processed_files_count} input files and saved a total of {total_chunks_saved} chunk files.")