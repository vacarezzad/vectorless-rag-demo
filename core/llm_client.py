from typing import Optional


class LLMClient:
    PROVIDERS = {
        "OpenAI": {
            "icon": "🟢",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        },
        "Anthropic (Claude)": {
            "icon": "🟠",
            "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        },
        "Google (Gemini)": {
            "icon": "🔵",
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        },
        "Mistral": {
            "icon": "🔴",
            "models": ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b"],
        },
    }

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model

    def call(self, prompt: str, system: Optional[str] = None) -> str:
        handlers = {
            "OpenAI": self._call_openai,
            "Anthropic (Claude)": self._call_anthropic,
            "Google (Gemini)": self._call_gemini,
            "Mistral": self._call_mistral,
        }
        if self.provider not in handlers:
            raise ValueError(f"Proveedor no soportado: {self.provider}")
        return handlers[self.provider](prompt, system)

    def _call_openai(self, prompt: str, system: Optional[str]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str, system: Optional[str]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text.strip()

    def _call_gemini(self, prompt: str, system: Optional[str]) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(full_prompt)
        return resp.text.strip()

    def _call_mistral(self, prompt: str, system: Optional[str]) -> str:
        from mistralai import Mistral

        client = Mistral(api_key=self.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.complete(
            model=self.model, messages=messages, temperature=0
        )
        return resp.choices[0].message.content.strip()
