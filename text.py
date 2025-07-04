from pymilvus import MilvusClient, model

client = MilvusClient("demo.db")

if client.has_collection(collection_name="demo"):
    client.drop_collection(collection_name="demo")
client.create_collection(collection_name="demo", dimension=768)


docs = [
    "Artificial intelligence(AI) was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

embedding_fn = model.DefaultEmbeddingFunction()
vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="demo", data=data)
print("insert result:", res)

query_vectors = embedding_fn.encode_queries(["When was artificial intelligence founded?"])

res = client.search(
    collection_name="demo",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print("search result:", res)
