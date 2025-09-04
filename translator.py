import modal
from typing import Dict
import json

app = modal.App("hunyuan-translator")

# Create image with all dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.1.0",
    "transformers==4.36.0",
    "sentencepiece==0.1.99",
    "accelerate==0.25.0",
    "bitsandbytes==0.41.3",
    "fastapi",
    "pydantic"
)

@app.cls(
    image=image,
    gpu=modal.gpu.A10G(),  # Need bigger GPU for 7B model
    memory=32768,  # 32GB RAM
    container_idle_timeout=600,  # Keep warm for 10 minutes
    timeout=300,  # 5 minute timeout for long texts
)
class HunyuanTranslator:
    @modal.enter()
    def setup(self):
        """Load the Hunyuan-MT-7B model on container startup"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🔄 Loading Hunyuan-MT-7B model...")
        model_name = "tencent/Hunyuan-MT-7B"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load model with 8-bit quantization to fit in memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,  # Use 8-bit quantization
            trust_remote_code=True
        )
        
        print("✅ Hunyuan-MT-7B model loaded successfully!")
        
        # Supported languages for translation to English
        self.supported_languages = {
            "chinese": "Chinese",
            "spanish": "Spanish", 
            "french": "French",
            "german": "German",
            "russian": "Russian",
            "japanese": "Japanese",
            "korean": "Korean",
            "arabic": "Arabic",
            "portuguese": "Portuguese",
            "italian": "Italian",
            "dutch": "Dutch",
            "turkish": "Turkish",
            "polish": "Polish",
            "swedish": "Swedish",
            "hindi": "Hindi",
            "vietnamese": "Vietnamese",
            "thai": "Thai",
            "indonesian": "Indonesian"
        }
    
    def format_translation_prompt(self, text: str, source_lang: str) -> str:
        """Format the prompt for Hunyuan-MT translation to English"""
        # Hunyuan-MT uses specific prompt format for translation
        prompt = f"""<|im_start|>system
You are a professional translator. Translate the following {source_lang} text to English accurately and naturally.
<|im_end|>
<|im_start|>user
Translate this {source_lang} text to English:

{text}
<|im_end|>
<|im_start|>assistant
Here is the English translation:

"""
        return prompt
    
    @modal.web_endpoint(method="POST")
    def translate(self, request: Dict) -> Dict:
        """Main translation endpoint for long-form text"""
        import torch
        
        text = request.get("text", "").strip()
        source_language = request.get("source_language", "chinese").lower()
        max_length = request.get("max_length", 4000)  # Max output length
        
        if not text:
            return {
                "status": "error",
                "error": "No text provided"
            }
        
        if len(text) > 10000:  # Limit input to 10k chars
            return {
                "status": "error", 
                "error": "Text too long. Maximum 10,000 characters."
            }
        
        # Get full language name
        lang_name = self.supported_languages.get(source_language, "Chinese")
        
        try:
            # Format prompt for translation
            prompt = self.format_translation_prompt(text, lang_name)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=8192,
                truncation=True
            ).to(self.model.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,  # Lower temp for accurate translation
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode the translation
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the translation part (after "Here is the English translation:")
            if "Here is the English translation:" in full_output:
                translation = full_output.split("Here is the English translation:")[-1].strip()
            else:
                # Fallback: remove the prompt part
                translation = full_output[len(prompt):].strip()
            
            # Clean up any remaining tokens
            translation = translation.replace("<|im_end|>", "").strip()
            
            return {
                "status": "success",
                "original_text": text,
                "translation": translation,
                "source_language": source_language,
                "target_language": "english",
                "model": "Hunyuan-MT-7B",
                "original_length": len(text),
                "translation_length": len(translation)
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Translation failed"
            }
    
    @modal.web_endpoint(method="POST")
    def translate_batch(self, request: Dict) -> Dict:
        """Batch translation endpoint for multiple texts"""
        texts = request.get("texts", [])
        source_language = request.get("source_language", "chinese").lower()
        
        if not texts:
            return {
                "status": "error",
                "error": "No texts provided"
            }
        
        if len(texts) > 10:
            return {
                "status": "error",
                "error": "Maximum 10 texts per batch"
            }
        
        results = []
        for text in texts:
            result = self.translate({
                "text": text,
                "source_language": source_language
            })
            results.append(result)
        
        return {
            "status": "success",
            "translations": results,
            "total": len(results)
        }
    
    @modal.web_endpoint(method="GET")
    def health(self) -> Dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "Hunyuan-MT-7B",
            "gpu": "A10G",
            "ready": True,
            "capabilities": {
                "max_input_length": 10000,
                "max_output_length": 4000,
                "batch_support": True,
                "languages": list(self.supported_languages.keys())
            }
        }
    
    @modal.web_endpoint(method="GET") 
    def languages(self) -> Dict:
        """Get supported languages"""
        return {
            "supported_source_languages": self.supported_languages,
            "target_language": "english",
            "total_languages": len(self.supported_languages),
            "note": "All translations are to English only"
        }

# Optional: Scheduled function to keep the model warm
@app.function(schedule=modal.Period(minutes=5))
def keep_warm():
    """Keep the container warm to avoid cold starts"""
    print("Keeping container warm...")
    return {"status": "warm"}
