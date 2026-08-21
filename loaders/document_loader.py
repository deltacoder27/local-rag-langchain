from langchain_community.document_loaders import TextLoader, DirectoryLoader


def load_documents():
    loader = DirectoryLoader(
    "documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)
    docs = loader.load()
    return docs

def clean_text(doc):
    cleaned_lines = []
    
    for line in doc.page_content.splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    return text


