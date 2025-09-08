    def setup_llm(self):
        """Setup Language Model based on configuration with minimal error handling"""
        provider = self.config['model_provider']['default']
        
        if provider == "openai":
            model_config = self.config['model_provider']['openai']
            self.llm = ChatOpenAI(
                model=model_config['model'],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens']
            )
        elif provider == "groq":
            model_config = self.config['model_provider']['groq']
            
            # Try primary model first - minimal changes to preserve original behavior
            try:
                print(f"🔧 Connecting to: {model_config['model']}")
                self.llm = ChatGroq(
                    model=model_config['model'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens']
                )
                print(f"✅ Connected to {model_config['model']}")
                
            except Exception as e:
                # If primary model fails, only try fallback if explicitly specified
                fallback_model = model_config.get('fallback_model')
                if fallback_model:
                    print(f"⚠️ Primary model failed, trying fallback: {fallback_model}")
                    self.llm = ChatGroq(
                        model=fallback_model,
                        temperature=model_config['temperature'],
                        max_tokens=model_config['max_tokens']
                    )
                    print(f"✅ Connected to fallback: {fallback_model}")
                else:
                    # No fallback specified, raise original error to maintain original behavior
                    raise e
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
