from langchain_community.document_loaders import DirectoryLoader
directory = '/home/kartik/books'

def load_docs(directory):
    loader= DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)
len(documents)

