# Created by David at 28.02.2023
# Project name szpj_semestralka
import codecs
import os
import pickle
import re
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import config
from config import model_transformer
from config import num_of_cores

_re_word_boundaries = re.compile(r'\b')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.parser()
        self.tfid = None

    def parser(self):
        f = codecs.open(self.path, 'r', 'utf-8')
        document = BeautifulSoup(f.read(),features="html.parser").get_text()
        text = "".join([s for s in document.strip().splitlines(True) if s.strip()])
        lines = text.splitlines()
        pattern = re.compile("^[0-9\s]+$")
        new_lines = []
        for line in lines:
            if not pattern.match(line):
                new_lines.append(line)
        self.sentence = " ".join(new_lines)

    def set_embeding(self, model):

        x = len(_re_word_boundaries.findall(self.sentence)) >> 1
        if x > model.max_seq_length:
            print(self.sentence)
        self.embeding = model.encode(self.sentence, convert_to_tensor=True)

    def compare_cosine(self, vec1):
        x = util.cos_sim(vec1, self.embeding)
        # x = np.dot(self.embeding,vec1)
        return x


def get_file_list(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))
    return filelist


def multicore(lst):
    model = SentenceTransformer(model_transformer)
    data_articles = []
    for i in lst:
        data_articles.append(DataLoader(i))
        data_articles[-1].set_embeding(model)
    return data_articles


def load_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    doc_list = []
    for doc_elem in root.findall('DOC'):
        docno = doc_elem.find('DOCNO').text.strip()
        doc_text = doc_elem.text.strip()
        doc_list.append(f'<DOC>\n<DOCNO>{docno}</DOCNO>\n\n{doc_text}\n</DOC>')
    return doc_list


def run_load(path):
    file_lst = get_file_list(Path(path))
    print(f"Num of files {len(file_lst)}")
    data_articles = []

    chunk_size = 50
    pool = Pool(processes=num_of_cores)

    chunks = [file_lst[x:x + chunk_size] for x in range(0, len(file_lst), chunk_size)]
    results = []

    for result in tqdm(pool.imap_unordered(multicore, chunks), total=len(chunks)):
        results.append(result)

    for i in results:
        data_articles.extend(i)

    sentences = []

    for i in data_articles:
        sentences.append(i.sentence)

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(sentences)

    for index, i in enumerate(doc_vectors):
        data_articles[index].tfid = i

    return data_articles, vectorizer


if __name__ == "__main__":
    file_lst = get_file_list(Path("SZPJ_SP1_collection/documents"))
    data_articles = []
    model = SentenceTransformer(model_transformer)

    chunk_size = 10
    pool = Pool(processes=config.num_of_cores)

    chunks = [file_lst[x:x + chunk_size] for x in range(0, len(file_lst), chunk_size)]
    results = []

    for result in tqdm(pool.imap_unordered(multicore, chunks), total=len(chunks)):
        results.append(result)

    for i in results:
        data_articles.extend(i)

    query = """ I am interested in articles written either by Prieve or Udo Pooch Prieve, B. Pooch, U."""

    sentences = []

    for i in data_articles:
        sentences.append(i.sentence)

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(sentences)

    for index, i in enumerate(doc_vectors):
        data_articles[index].tfid = i

    with open('embedings_data.pkl', 'wb') as f:
        pickle.dump([data_articles, vectorizer], f)

    print("Embedings done")
