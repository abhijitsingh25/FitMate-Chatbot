from pinecone import Pinecone

pc = Pinecone(api_key = "pcsk_3ryyUH_PSXqouQsYgpTRr78Ynme7CSJS4ocVL8ExscaJk4FMerw2RRmnhspQb2ryW6DYWf")
index = pc.Index("quickstart")