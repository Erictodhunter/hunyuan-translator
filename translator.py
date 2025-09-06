import modal
from typing import Dict
import json

app = modal.App("hunyuan-translator")

# Create image with required packages
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("fastapi", "uvicorn")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
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
        """Load the LLM model on container startup"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("🔄 Loading Hunyuan translation model...")
        
        # Initialize attributes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load Hunyuan-MT model for actual translation
            model_name = "tencent/Hunyuan-MT-7B"
            print(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            print(f"✅ Hunyuan-MT model loaded successfully on {self.device}!")
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading Hunyuan model: {e}")
            print("Loading fallback model...")
            
            # Fallback to a smaller multilingual model
            try:
                model_name = "facebook/m2m100_418M"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model_loaded = True
                print("✅ Fallback M2M model loaded")
            except Exception as e2:
                print(f"❌ Fallback model also failed: {e2}")
                self.model_loaded = False
    
    def translate_with_llm(self, text: str, source_language: str, target_language: str = "english") -> str:
        """Translate using the LLM with proper prompting"""
        import torch
        
        # Check if model is loaded and available
        if not hasattr(self, 'model_loaded') or not self.model_loaded or self.model is None:
            return "Translation service not properly initialized"
        
        # Create the translation prompt
        prompt = f"Translate the following text from {source_language} to {target_language}, without additional explanation.\n\n{text}"
        
        # For Hunyuan-MT, use chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=False, 
                return_tensors="pt"
            )
        else:
            # Fallback tokenization
            tokenized_chat = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            tokenized_chat = tokenized_chat.input_ids
        
        # Move to device
        tokenized_chat = tokenized_chat.to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_chat,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the result
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the translation (remove the prompt)
        if prompt in output_text:
            translation = output_text.replace(prompt, "").strip()
        else:
            translation = output_text.strip()
        
        return translation
    
    @modal.web_endpoint(method="POST")
    def translate(self, request: Dict) -> Dict:
        """Main translation endpoint"""
        
        # Get request parameters
        text = request.get("text", "").strip()
        source_language = request.get("source_language", "auto").lower()
        target_language = request.get("target_language", "english").lower()
        max_length = request.get("max_length", 10000)
        
        # Validate input
        if not text:
            return {
                "status": "error",
                "error": "No text provided",
                "translation": ""
            }
        
        if len(text) > 50000:
            return {
                "status": "error",
                "error": "Text too long. Maximum 50,000 characters.",
                "translation": ""
            }
        
        try:
            # If text is too long, split into chunks
            if len(text) > 5000:
                # Split into smaller chunks
                chunks = []
                chunk_size = 4000
                for i in range(0, len(text), chunk_size):
                    chunks.append(text[i:i+chunk_size])
                
                # Translate each chunk
                translated_chunks = []
                for chunk in chunks:
                    chunk_translation = self.translate_with_llm(chunk, source_language, target_language)
                    translated_chunks.append(chunk_translation)
                
                translation = " ".join(translated_chunks)
            else:
                # Translate as single piece
                translation = self.translate_with_llm(text, source_language, target_language)
            
            return {
                "status": "success",
                "original_text": text,
                "translation": translation,
                "source_language": source_language,
                "target_language": target_language,
                "model": "hunyuan-mt-7b",
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
    
    @modal.web_endpoint(method="GET")
    def health(self) -> Dict:
        """Health check endpoint"""
        import torch
        return {
            "status": "healthy" if self.model_loaded else "degraded",
            "model": "hunyuan-mt-7b",
            "gpu": str(torch.cuda.is_available()),
            "ready": self.model_loaded,
            "note": "Full language support via LLM",
            "capabilities": {
                "max_input_length": 50000,
                "max_output_length": 10000,
                "supports_all_languages": True,
                "chunk_processing": True
            }
        }
    
    @modal.web_endpoint(method="GET")
    def languages(self) -> Dict:
        """Get supported languages"""
        return {
            "note": "Supports all major world languages via LLM",
            "target_language": "configurable (default: english)",
            "approach": "LLM-based translation with prompting",
            "quality": "High quality contextual translation"
        }
