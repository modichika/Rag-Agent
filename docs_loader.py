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
print(docs[0].page_content[:500]) # string slicing operation
print(type(docs))

