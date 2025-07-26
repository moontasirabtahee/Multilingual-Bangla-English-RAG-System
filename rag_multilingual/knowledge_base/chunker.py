from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(merged_text, chunk_size=350, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return splitter.split_text(merged_text)