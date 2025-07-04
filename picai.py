COLLECTION_NAME = 'image_search'  # Collection name
DIMENSION = 1024  # Embedding vector size in this example
TOP_K = 3

from pymilvus import FieldSchema, CollectionSchema, DataType, connections, Collection, utility

connections.connect(uri="image.db")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='filepath', dtype=DataType.VARCHAR, max_length=200),  # VARCHARS need a maximum length, so for this example they are set to 200 characters
    FieldSchema(name='image_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
'metric_type':'L2',
'index_type':"IVF_FLAT",
'params':{'nlist': 16384}
}
collection.create_index(field_name="image_embedding", index_params=index_params)
collection.load()

import glob

paths = glob.glob('./wallpaper/*.png', recursive=True)
print(len(paths))

import base64
import dashscope
from http import HTTPStatus

def embed_image(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        image_format = "png"  # 根据实际情况修改，比如jpg、bmp 等
        image_data = f"data:image/{image_format};base64,{data}"
        print("image data len:", len(image_data))
        inputs = [{'image': image_data}]

        try:
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=inputs,
                request_timeout=10,
                api_key="<your_api_key>"  # Replace with your actual API key
                                          # or use environment variable DASH_SCOPE_API_KEY
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output["embeddings"][0]["embedding"]
            else:
                return None
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return None

embeds = []
files = []
for path in paths:
    embed = embed_image(path)
    if embed is None:
        continue
    embeds.append(embed)
    files.append(path)

print(len(embeds), len(files))
collection.insert([files, embeds])
collection.flush()


import time
from matplotlib import pyplot as plt
from PIL import Image

search_paths = paths[:3]
embeds = []
files = []
for path in search_paths:
    embed = embed_image(path)
    if embed is None:
        continue
    embeds.append(embed)
    files.append(path)
print(len(embeds), len(files))

start = time.time()
res = collection.search(embeds, anns_field='image_embedding', param={'nprobe': 128}, limit=TOP_K, output_fields=['filepath'])
finish = time.time()
print(res)
f, axarr = plt.subplots(len(files), TOP_K + 1, figsize=(20, 10), squeeze=False)

for hits_i, hits in enumerate(res):
    axarr[hits_i][0].imshow(Image.open(files[hits_i]))
    axarr[hits_i][0].set_axis_off()
    axarr[hits_i][0].set_title('Search Time: ' + str(finish - start))
    for hit_i, hit in enumerate(hits):
        axarr[hits_i][hit_i + 1].imshow(Image.open(hit.entity.get('filepath')))
        axarr[hits_i][hit_i + 1].set_axis_off()
        axarr[hits_i][hit_i + 1].set_title('Distance: ' + str(hit.distance))

plt.savefig('search_result.png')
