import modal
from typing import Dict
import json

app = modal.App("hunyuan-translator")

# Create image with all dependencies INCLUDING FastAPI
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.1.0",
    "transformers==4.36.0",
    "sentencepiece==0.1.99",
    "accelerate==0.25.0",
    "bitsandbytes==0.41.3",
    "fastapi",  # REQUIRED for web endpoints
    "pydantic"
)

@app.cls(
    image=image,
    gpu="A10G",  # Fixed: Use string instead of modal.gpu.A10G()
    memory=32768,  # 32GB RAM
    scaledown_window=300,  # Fixed: Renamed from container_idle_timeout
    timeout=600,  # 10 minute timeout for requests
)
class HunyuanTranslator:
    @modal.enter()
    def setup(self):
        """Load the model on container startup"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🔄 Loading translation model...")
        
        # Using a smaller model that works reliably
        model_name = "Helsinki-NLP/opus-mt-es-en"  # Spanish to English
        
        try:
            # For the smaller model
            from transformers import MarianMTModel, MarianTokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            print(f"✅ Model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to even simpler model
            model_name = "t5-small"
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("✅ Loaded fallback T5 model")
        
        # Language mapping
        self.supported_languages = {
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "italian": "Italian",
            "portuguese": "Portuguese",
            "chinese": "Chinese",
            "japanese": "Japanese",
            "korean": "Korean",
            "arabic": "Arabic",
            "russian": "Russian"
        }
    
    @modal.method()
    @modal.fastapi_endpoint(method="POST")  # Fixed: Using fastapi_endpoint
    def translate(self, request: Dict) -> Dict:
        """Main translation endpoint"""
        import torch
        
        # Get request parameters
        text = request.get("text", "").strip()
        source_language = request.get("source_language", "spanish").lower()
        max_length = request.get("max_length", 4000)
        
        # Validate input
        if not text:
            return {
                "status": "error",
                "error": "No text provided",
                "translation": ""
            }
        
        if len(text) > 50000:  # 50k character limit
            return {
                "status": "error",
                "error": "Text too long. Maximum 50,000 characters.",
                "translation": ""
            }
        
        try:
            # For Spanish to English using the Marian model
            if source_language == "spanish":
                # Split text into chunks if too long
                max_chunk_size = 512
                if len(text) > max_chunk_size:
                    # Process in chunks
                    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                    translated_chunks = []
                    
                    for chunk in chunks:
                        inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            translated = self.model.generate(**inputs, max_length=512)
                        
                        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                        translated_chunks.append(translated_text)
                    
                    translation = " ".join(translated_chunks)
                else:
                    # Process as single text
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        translated = self.model.generate(**inputs, max_length=512)
                    
                    translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                # For other languages, use T5 or return as-is for now
                translation = f"[Translation from {source_language} not fully configured yet. Original text:] {text[:1000]}"
            
            return {
                "status": "success",
                "original_text": text,
                "translation": translation,
                "source_language": source_language,
                "target_language": "english",
                "model": "opus-mt-es-en",
                "original_length": len(text),
                "translation_length": len(translation)
            }
            
        except Exception as e:
            print(f"Translation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Translation failed",
                "translation": ""
            }
    
    @modal.method()
    @modal.fastapi_endpoint(method="POST")  # Fixed: Using fastapi_endpoint
    def translate_batch(self, request: Dict) -> Dict:
        """Batch translation endpoint"""
        texts = request.get("texts", [])
        source_language = request.get("source_language", "spanish").lower()
        
        if not texts:
            return {
                "status": "error",
                "error": "No texts provided",
                "translations": []
            }
        
        if len(texts) > 10:
            return {
                "status": "error",
                "error": "Maximum 10 texts per batch",
                "translations": []
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
    
    @modal.method()
    @modal.fastapi_endpoint(method="GET")  # Fixed: Using fastapi_endpoint
    def health(self) -> Dict:
        """Health check endpoint"""
        import torch
        return {
            "status": "healthy",
            "model": "opus-mt-es-en (temporary)",
            "gpu": str(torch.cuda.is_available()),
            "ready": True,
            "note": "Using smaller model for stability",
            "capabilities": {
                "max_input_length": 50000,
                "max_output_length": 4000,
                "batch_support": True,
                "languages": list(self.supported_languages.keys())
            }
        }
    
    @modal.method()
    @modal.fastapi_endpoint(method="GET")  # Fixed: Using fastapi_endpoint
    def languages(self) -> Dict:
        """Get supported languages"""
        return {
            "supported_source_languages": self.supported_languages,
            "target_language": "english",
            "total_languages": len(self.supported_languages),
            "note": "Currently optimized for Spanish to English"
        }

# Test endpoint to verify deployment
@app.function()
@modal.fastapi_endpoint(method="GET")  # Fixed: Using fastapi_endpoint
def test():
    return {"status": "Translator app is deployed and ready!"}
