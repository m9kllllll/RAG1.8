from __future__ import annotations
import base64, os
from pathlib import Path
import httpx

class ImageCaptioner:
    """Optional Ollama vision captioner for diagrams, figures and scanned tables."""
    def __init__(self, base_url=None, model=None, timeout=120.0):
        self.base_url=(base_url or os.getenv("OLLAMA_BASE_URL","http://localhost:11434")).rstrip("/")
        self.model=model or os.getenv("VISION_MODEL","gemma3:4b"); self.timeout=timeout
    def caption_bytes(self,image_bytes:bytes,prompt=None)->str:
        prompt=prompt or "Describe this academic document image accurately. Extract visible text, labels, chart/table meaning and relationships. Do not invent missing information."
        payload={"model":self.model,"prompt":prompt,"images":[base64.b64encode(image_bytes).decode("ascii")],"stream":False}
        r=httpx.post(f"{self.base_url}/api/generate",json=payload,timeout=self.timeout); r.raise_for_status()
        return str(r.json().get("response","")).strip()
    def caption_file(self,path:str|Path,prompt=None)->str:
        return self.caption_bytes(Path(path).read_bytes(),prompt)
    def to_searchable_text(self,caption,image_path,page=None)->str:
        return f"[IMAGE_CAPTION]\nImage: {image_path}\nLocation: page {page or 'unknown'}\nCaption: {caption}\n[/IMAGE_CAPTION]"
