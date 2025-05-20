from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from fsspec import AbstractFileSystem
from html2text import html2text
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs
from llama_index.core.schema import Document


class HTMLFileMarkdownReader(BaseReader):
    """Reads HTML files into markdown using html2text package."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        fs = fs or get_default_fs()

        with fs.open(file, "rb") as fp:
            soup = BeautifulSoup(fp.read(), "html.parser")
            title = next(iter(soup.find_all("title")), None)
            metadata = {
                "file_name": file.name[:256],
                "page_title": str(title)[:256],
            }
            if extra_info:
                metadata.update(extra_info)
        with fs.open(file, "r") as fp:
            text = fp.read()
            text = html2text(text)
        return [Document(text=text, extra_info=metadata)]
