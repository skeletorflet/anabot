import logging
import os
import random
import re

# Setup simpler logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_ITER = 4

# Copied from bot.py
def process_dynamic_keywords(prompt):
    """
    Scans resources folder for txt files.
    If filename matches a word in prompt using regex word boundary, replaces it with {line|line|...}
    using 5-10 random lines.
    Uses re.sub to ensure each occurrence gets a FRESH random sample.
    """
    resources_dir = "resources"
    if not os.path.exists(resources_dir):
        return prompt
        
    for filename in os.listdir(resources_dir):
        if filename.endswith(".txt"):
            keyword = os.path.splitext(filename)[0]
            
            # Regex to find keyword as a whole word (so f_anime doesn't match f_anime_2)
            # Escaping keyword just in case it has special chars
            pattern = r'\b' + re.escape(keyword) + r'\b'
            
            if re.search(pattern, prompt):
                file_path = os.path.join(resources_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Handle case where file might be one big line or mixed newlines
                        # Also replace | with , to avoid breaking syntax inside options
                        raw_lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                        # Sanitize and unique
                        lines = list(set(line.strip().replace('|', ',') for line in raw_lines if line.strip()))
                    
                    if not lines:
                        continue
                        
                    def replace_callback(match):
                        # Use a random subset of lines to prevent prompt explosion
                        # User requested using BASE_ITER (5) count
                        unique_options = list(lines)
                        random.shuffle(unique_options)
                        selected = unique_options[:BASE_ITER]
                        
                        # Construct dynamic prompt syntax: {a|b|c}
                        return "{" + "|".join(selected) + "}"
                    
                    # Replace in prompt
                    new_prompt = re.sub(pattern, replace_callback, prompt)
                    
                    if new_prompt != prompt:
                         logger.info(f"Replaced keyword '{keyword}' with {len(lines)} available options.")
                         prompt = new_prompt
                    
                except Exception as e:
                    logger.error(f"Error processing resource {filename}: {e}")
                    
    return prompt

# Test function
def test_randomization():
    prompt = "generate f_anime details"
    print(f"Original Prompt: {prompt}")
    
    # Ensure we are in the right directory or mock resources dir presence if needed
    # bot.py expects 'resources' dir in cwd
    if not os.path.exists("resources"):
        print("Creating mock resources for test...")
        os.makedirs("resources", exist_ok=True)
        with open("resources/f_anime.txt", "w") as f:
            for i in range(10):
                f.write(f"line_{i}\n")

    expanded = process_dynamic_keywords(prompt)
    print(f"Expanded Prompt: {expanded}")

if __name__ == "__main__":
    test_randomization()
