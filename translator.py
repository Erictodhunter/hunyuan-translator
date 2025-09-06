import modal
from typing import Dict
import json

app = modal.App("hunyuan-translator")

# Create image with FastAPI installed FIRST
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("fastapi", "uvicorn")  # Install FastAPI and uvicorn FIRST
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "sentencepiece==0.1.99",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        "pydantic"
    )
)

@app.cls(
    image=image,
    gpu="A10G",
    memory=32768,
    scaledown_window=300,
    timeout=600,
)
class HunyuanTranslator:
    def __enter__(self):
        """Load the model on container startup (Modal v1.0 style)"""
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        
        print("🔄 Loading translation models...")
        
        # Load multiple models for different languages
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Language to model mapping
        self.language_models = {
            "spanish": "Helsinki-NLP/opus-mt-es-en",
            "french": "Helsinki-NLP/opus-mt-fr-en", 
            "german": "Helsinki-NLP/opus-mt-de-en",
            "italian": "Helsinki-NLP/opus-mt-it-en",
            "portuguese": "Helsinki-NLP/opus-mt-pt-en"
        }
        
        # Load models for supported languages
        for lang, model_name in self.language_models.items():
            try:
                print(f"Loading {lang} model: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model = model.to(self.device)
                
                self.tokenizers[lang] = tokenizer
                self.models[lang] = model
                print(f"✅ {lang} model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading {lang} model: {e}")
        
        # Fallback model
        if not self.models:
            print("Loading fallback model...")
            try:
                self.tokenizers["fallback"] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
                self.models["fallback"] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en").to(self.device)
                print("✅ Fallback model loaded")
            except Exception as e:
                print(f"❌ Fallback model failed: {e}")
        
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
    
    def translate_text(self, text: str, source_language: str) -> str:
        """Translate text using appropriate model"""
        import torch
        
        # Check if we have a model for this language
        if source_language in self.models:
            model = self.models[source_language]
            tokenizer = self.tokenizers[source_language]
        elif "fallback" in self.models:
            model = self.models["fallback"]
            tokenizer = self.tokenizers["fallback"]
            print(f"Using fallback model for {source_language}")
        else:
            return f"No translation model available for {source_language}"
        
        # Split text into chunks if too long
        max_chunk_size = 512
        if len(text) > max_chunk_size:
            # Process in chunks
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(translated_text)
            
            return " ".join(translated_chunks)
        else:
            # Process as single text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    @modal.web_endpoint(method="POST")
    def translate(self, request: Dict) -> Dict:
        """Main translation endpoint"""
        
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
            # Translate the text
            translation = self.translate_text(text, source_language)
            
            # Determine which model was used
            model_used = self.language_models.get(source_language, "fallback-opus-mt-es-en")
            
            return {
                "status": "success",
                "original_text": text,
                "translation": translation,
                "source_language": source_language,
                "target_language": "english",
                "model": model_used,
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
    
    @modal.web_endpoint(method="POST")
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
    
    @modal.web_endpoint(method="GET")
    def health(self) -> Dict:
        """Health check endpoint"""
        import torch
        return {
            "status": "healthy",
            "models_loaded": list(self.models.keys()),
            "gpu": str(torch.cuda.is_available()),
            "ready": True,
            "note": "Multi-language translation support",
            "capabilities": {
                "max_input_length": 50000,
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
            "models_available": list(self.models.keys()),
            "note": "Supports French, Spanish, German, Italian, Portuguese"
        }
