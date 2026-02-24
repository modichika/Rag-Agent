# Injection Engine data->document with same format
import bs4
from langchain_community.document_loaders import WebBaseLoader

#Only keep post title, headers, and content from full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader( # WebBaseLoader is a class designed to handle multiple websites at once. The "Machine" itself. Loader uses the WebBaseLoader Object
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), # why a , after ""? it's a tuple otherwise python will think that it's just a string. A tuple with one element needs a coma after the element. (Where to go?)
    bs_kwargs={"parse_only": bs4_strainer}, # BeautifulSoup keyword arguments (What to look for?)

) # loader is an engine object.

docs = loader.load() # triggers actual netwrok request, return Document objects, each object contains - page_content, metadata(a dictionary containing "data about the"). Returns a List[] not a tuple()

assert len(docs) == 1 # ensures only one document is loaded
print(f"Total characters: {len(docs[0].page_content)}")
#print(docs[0].page_content[:500]) # string slicing operation
#print(type(docs))

# For Splitting the document into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter( # the recursive one cuts the whole blog into smaller after paras, new lines, then individual chars. Not through any char("A-gent"). text_splitter is an object not a list, tuple whose length cannot be taken out 
    chunk_size=1000, # size of each chunk in chars.
    chunk_overlap=200, # from the end of chunk 1 it will appear in the beginning of chunk 2.
    add_start_index=True, # a boolean flag to calculate the exact char position where each chunk begins i.e, the start char, it adds the start_index to the metadata dictionary of each chunk. A kind of bookmark to tell which chunk we are on
)

all_splits = text_splitter.split_documents(docs) # new list of smaller document object from the docs by looping through the docs.

print(f"Split blog post into {len(all_splits)} sub-documents.") # we have 63 sub docs chunked each has 1000 chars.
#print(text_splitter._chunk_size, text_splitter._chunk_overlap, text_splitter._add_start_index)
#print(all_splits[0].metadata) # prints 2 everytime coz langchain attatches two specific labels to every single chunk : 1. source, 2. start_index.

#print(repr(docs[0].page_content[:8]))

# Note if the print of len of all_splits is 1 then our logic has failed.



