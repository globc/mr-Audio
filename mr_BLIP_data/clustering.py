import os
import json

def save_json(content, save_path):
    # if no such directory, create one
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    

if __name__ == "__main__":
    # Measure against overfitting: Ambiguity by clustering
    # From: https://github.com/amazon-science/auto-cot/
    ann_root = "/home/atuin/g102ea/g102ea15/mr-Audio/mr_BLIP_data/charades_sta_annotations"
    avg_num_equal_queries = 8
    num_clusters = round(11166 * 1.0/avg_num_equal_queries)
    new_train_path = ann_root + '/lavis/new_train.json'

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import random

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    data = load_json(new_train_path)
    corpus = [sample["query"] for sample in data]

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = encoder.encode(corpus)

    # KMeans clustering
    clustering_model = KMeans(n_clusters=num_clusters, random_state=seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_


    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_idx = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    center_queries = []
    for i in range(num_clusters):
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        min_idx = top_min_dist[0][0]
        center_queries.append(corpus[clustered_idx[i][min_idx]])

    ambiguous_data = []
    for sample, cluster_label in zip(data, cluster_assignment):
        sample["query"] = center_queries[cluster_label]
        ambiguous_data.append(sample)

    save_json(ambiguous_data, ann_root + '/lavis/ambiguous_train_' + str(avg_num_equal_queries) + '.json')