import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

import config
from config import model_transformer
from load_data import run_load


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
if __name__ == '__main__':
    model = SentenceTransformer(model_transformer)

    from multiprocessing import set_start_method

    set_start_method("spawn")

    data_articles, vectorizer = run_load(Path(config.path))
    question_lst = load_xml_file(Path(config.path_query))

    file = open(config.output_name, 'w+')

    for docno, doc_text in tqdm(question_lst):
        doc_text_embeding = model.encode(doc_text, convert_to_tensor=True)
        idf_embeding = vectorizer.transform([doc_text])
        ranking = []
        rank = {}
        for i in data_articles:
            x = i.compare_cosine(doc_text_embeding)
            cosine_similarities = linear_kernel(i.tfid, idf_embeding)

            file_name, _ = os.path.splitext(os.path.basename(i.path))
            rank[file_name] = x[0][0].item() + cosine_similarities[0, 0]
            # rank[file_name] = x
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        for i in rank[0:100]:
            file.write(f"{docno}\t{i[0]}\t{i[1]}\n")

    file.close()

    print("Done")
