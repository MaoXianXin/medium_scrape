# translate_example.py
import logging
import os
import json # Added for saving ChatML
from pathlib import Path # Added for easier path handling
from flask_endpoint.dialog_module.utils import create_custom_llm
from flask_endpoint.dialog_module.base import OneTimeDialogModule

"""
python -m flask_endpoint.data_preparation.translate_example
"""

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the directory containing the input .txt files
INPUT_TEXT_DIR = Path("/home/mao/datasets/machine_translation/en_txt_chunks") # <<< Define your input directory path here
OUTPUT_CHATML_DIR = Path("/home/mao/datasets/machine_translation/en_txt_chunks_chatml") # <<< Define your output directory path here

# Create output directory if it doesn't exist
OUTPUT_CHATML_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define the revised prompt template content
#    (You could also save this to a file and use read_template_from_file)
translation_prompt = """
你是一位专业的技术文档翻译专家，精通中英双语，特别擅长技术文档的翻译。请将以下英文技术文档翻译成中文，要求：

1. 保持技术术语的准确性和一致性
2. 确保技术概念表达清晰准确
3. 译文要符合中文技术文档的表达习惯
4. 保持原文的技术严谨性
5. 对于专业术语，如果已有通用中文译法，请使用通用译法
6. 对于没有通用译法的专业术语，请给出准确且易于理解的中文翻译

英文原文：
{input_text}

请提供翻译结果，并标注任何需要特别说明的术语处理。
"""

def save_chatml(input_text, output_text, output_filepath):
    """Saves the conversation in ChatML format."""
    chatml_data = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(chatml_data, f, ensure_ascii=False, indent=2)
        logger.info(f"ChatML history saved to: {output_filepath}")
    except IOError as e:
        logger.error(f"Failed to save ChatML to {output_filepath}: {e}")

def main():
    try:
        # 3. Create the LLM instance using the utility function
        #    You can override defaults like model_name, temperature, etc. if needed
        #    Make sure the base_url and api_key are correctly configured for your LLM provider
        logger.info("Creating LLM instance...")
        llm = create_custom_llm(
            model_name="gemini-2.5-pro-exp-03-25",
            base_url="https://zzzzapi.com/v1",
            api_key="sk-UxCneocSvk83jPkSmDRyYZA2zLWiAX1Ds71JVK72IqH1DiR6"
        )
        logger.info("LLM instance created.")

        # 4. Create the OneTimeDialogModule instance for translation
        logger.info("Initializing translation module...")
        translator = OneTimeDialogModule(
            llm=llm,
            prompt_template=translation_prompt
            # max_retries can also be set here if needed
        )
        logger.info("Translation module initialized.")

        # 5. Find and process all .txt files in the input directory
        logger.info(f"Looking for .txt files in: {INPUT_TEXT_DIR}")
        txt_files = list(INPUT_TEXT_DIR.glob("*.txt"))

        if not txt_files:
            logger.warning(f"No .txt files found in {INPUT_TEXT_DIR}. Exiting.")
            print(f"Warning: No .txt files found in {INPUT_TEXT_DIR}")
            return

        for txt_file_path in txt_files:
            logger.info(f"--- Processing file: {txt_file_path.name} ---")
            try:
                # Read the English text from the file
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    english_text_to_translate = f.read()

                if not english_text_to_translate.strip():
                    logger.warning(f"File {txt_file_path.name} is empty. Skipping.")
                    continue

                # Process the translation
                logger.info(f"Starting translation for {txt_file_path.name}...")
                chinese_translation = translator.process(input_text=english_text_to_translate)
                logger.info(f"Translation process finished for {txt_file_path.name}.")

                if not chinese_translation:
                    logger.warning(f"Translation failed for {txt_file_path.name}. Skipping ChatML saving.")
                    print(f"Warning: Translation failed for {txt_file_path.name}. Skipping.")
                    continue

                # Define output ChatML filename
                output_filename = txt_file_path.stem + "_chatml.json"
                output_filepath = OUTPUT_CHATML_DIR / output_filename

                # Save the result in ChatML format
                save_chatml(english_text_to_translate, chinese_translation, output_filepath)

                # Optional: Print results to console as well
                print("-" * 20)
                print(f"File: {txt_file_path.name}")
                print("Original English Text:")
                print(english_text_to_translate) # Print snippet
                print("\nChinese Translation:")
                print(chinese_translation) # Print snippet
                print(f"ChatML saved to: {output_filepath}")
                print("-" * 20 + "\n")


            except FileNotFoundError:
                logger.error(f"File not found: {txt_file_path}. Skipping.")
                print(f"Error: File not found: {txt_file_path}")
            except IOError as e:
                logger.error(f"Error reading file {txt_file_path.name}: {e}. Skipping.")
                print(f"Error reading file {txt_file_path.name}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred processing {txt_file_path.name}: {e}", exc_info=True)
                print(f"An unexpected error occurred processing {txt_file_path.name}: {e}")


    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        print(f"Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during initialization: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()