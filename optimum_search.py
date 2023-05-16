import os
from pathlib import Path
import uuid

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from multiprocessing import Pool

import compute_score_fun
from compute_score_fun import score

import config
from config import model_transformer
from load_data import run_load
from compute_score_fun import score
from typing import Tuple


def load_xml_file(file_path):
    with open(file_path, 'r') as f:
        doc_list = []
        docno = None
        doc_text = ""
        for line in f:
            line = line.strip()
            if line.startswith('<DOCNO>'):
                docno = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
            elif line.startswith('</DOC>'):
                if docno is not None:
                    doc_list.append((docno, doc_text.strip()))
                    docno = None
                    doc_text = ""
            elif line.startswith('<DOC>') or line.startswith('</DOCNO>'):
                # Ignore start and end tags
                continue
            else:
                doc_text += line + " "
    return doc_list


# Press the green button in the gutter to run the script.
def main(params: Tuple[float, float, dict, list, dict]) -> Tuple[float, float, float]:
    Y1, Y2, doc_embeddings, data_articles, idf_embeddings = params

    filename = 'tmp/'+str(uuid.uuid4())

    file = open(filename, 'w+')

    for docno in doc_embeddings:
        doc_text_embeding = doc_embeddings[docno]
        idf_embeding = idf_embeddings[docno]
        rank = {}
        for i in data_articles:
            x = i.compare_cosine(doc_text_embeding)
            cosine_similarities = linear_kernel(i.tfid, idf_embeding)

            file_name, _ = os.path.splitext(os.path.basename(i.path))
            rank[file_name] = x*Y2 + cosine_similarities[0, 0]*Y1
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        for i in rank[0:100]:
            file.write(f"{docno}\t{i[0]}\t{i[1]}\n")

    file.close()
    score = compute_score_fun.score(filename)
    os.remove(filename)
    # Return the score along with the parameters that produced it
    return Y1, Y2, score


def grid_search() -> Tuple[float, float]:
    best_Y1, best_Y2 = None, None
    best_score = float('-inf')

    model = SentenceTransformer(model_transformer, device="cuda")

    data_articles, vectorizer = run_load(Path(config.path), model)
    question_lst = load_xml_file(Path(config.path_query))

    doc_embeddings = {docno: model.encode(doc_text, convert_to_tensor=False) for docno, doc_text in question_lst}
    idf_embeddings = {docno: vectorizer.transform([doc_text]) for docno, doc_text in question_lst}

    # Create a list of all parameters to try
    params = [(Y1 / 100, Y2 / 100, doc_embeddings, data_articles, idf_embeddings)
              for Y1 in range(10, 130, 1) for Y2 in range(0, 201, 10)]

    # Use multiprocessing to compute the score for each set of parameters
    with Pool(28) as p:
        results = p.map(main, params)

    # Find the best score and the corresponding parameters
    for Y1, Y2, score in results:
        if score > best_score:
            best_score = score
            best_Y1, best_Y2 = Y1, Y2

    return best_Y1, best_Y2

if __name__ == '__main__':
    Y1, Y2 = grid_search()
    print(f"Optimal values: Y1 = {Y1}, Y2 = {Y2}")


