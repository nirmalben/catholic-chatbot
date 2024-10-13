from langchain_community.document_loaders import JSONLoader
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from DocumentLoader import DocumentLoader

class JsonDocumentLoader(DocumentLoader):
  def _default_metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    return metadata

  def __init__(self, json_file_path: str, jq_schema='.[]', metadata_func=_default_metadata_func, content_key='text') -> None:
    super().__init__(json_file_path)
    self.jq_schema = jq_schema
    self.metadata_func = metadata_func
    self.content_key = content_key
    self._load()

  def _load(self) -> None:
    loader = JSONLoader(
      file_path=self.file_path,
      jq_schema=self.jq_schema,
      metadata_func=self.metadata_func,
      content_key=self.content_key,
      text_content=False
    )
    self.documents = loader.load()
