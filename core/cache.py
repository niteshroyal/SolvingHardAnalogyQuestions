import os
import json
import time
import logging
import rocksdb
import filelock
import resource
from relbert import RelBERT
from reasoning_with_vectors.conf import configuration


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class CustomRelBERT(RelBERT):
    def encode_word_pairs(self, word_pairs, parallel: bool = False):
        return super().encode_word_pairs(word_pairs, parallel=parallel)


class Cache:
    def __init__(self, read_only=False, db_path=configuration.rocksdb_path, lock_path=configuration.rocksdb_lock):
        current_soft_limit, current_hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft_limit = min(8192, current_hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, current_hard_limit))
        self.model = CustomRelBERT("relbert/relbert-roberta-large-nce-semeval2012-0-400")
        self.relbert_embeddings = b"relbert_embeddings:"
        self.paths_length_2 = b"paths_length_2:"
        self.paths_length_3 = b"paths_length_3:"
        self.db = None
        self.lock = None
        self.init_db(read_only, db_path, lock_path)

    def init_db(self, read_only, db_path, lock_path):
        # options = rocksdb.Options()
        # options.create_if_missing = True
        # table_options = rocksdb.BlockBasedTableFactory(
        #     filter_policy=rocksdb.BloomFilterPolicy(10),
        #     block_cache=rocksdb.LRUCache(32 * (1024 ** 3))
        # )
        # options.table_factory = table_options
        # options.compression = rocksdb.CompressionType.zstd_compression
        # options.target_file_size_base = 4 * 128 * 1024 * 1024
        # options.max_open_files = min(300000, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        # options.write_buffer_size = 128 * 1024 * 1024
        # self.db = rocksdb.DB(db_path, options, read_only=read_only)
        options = rocksdb.Options()
        options.create_if_missing = True
        options.max_open_files = min(300000, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        self.db = rocksdb.DB(db_path, options, read_only=read_only)
        self.lock = filelock.FileLock(lock_path)

    def set_stored_relbert_embds(self, key, value):
        key_str = json.dumps(key)
        value_str = json.dumps(value)
        with self.lock:
            self.db.put(self.relbert_embeddings + key_str.encode(), value_str.encode())

    def get_stored_relbert_embds(self, key):
        key_str = json.dumps(key)
        value = self.db.get(self.relbert_embeddings + key_str.encode())
        if value:
            return json.loads(value.decode())
        return None

    def check_stored_relbert_embds(self, key):
        key_str = json.dumps(key)
        value = self.db.get(self.relbert_embeddings + key_str.encode())
        return value is not None

    def all_stored_relbert_embds(self):
        it = self.db.iteritems()
        it.seek(self.relbert_embeddings)
        result = []
        for key, _ in it:
            key = key.decode()
            if key.startswith("relbert_embeddings:"):
                result.append(json.loads(key.split("relbert_embeddings:", 1)[1]))
            else:
                break
        return result

    def erase_relbert_embeddings(self, keys):
        for key in keys:
            key_str = json.dumps(key)
            self.db.delete(self.relbert_embeddings + key_str.encode())

    def store_related_concepts(self, concept1, concept2, related):
        key_str = json.dumps([concept1, concept2])
        value_str = json.dumps(related)
        with self.lock:
            self.db.put(self.paths_length_2 + key_str.encode(), value_str.encode())

    def get_store_related_concepts(self, concept1, concept2):
        key_str = json.dumps([concept1, concept2])
        value = self.db.get(self.paths_length_2 + key_str.encode())
        if value:
            return json.loads(value.decode())
        return None

    def delete_all_stored_related_concepts(self):
        with self.lock:
            it = self.db.iteritems()
            it.seek(self.paths_length_2)
            keys_to_delete = []
            for key, _ in it:
                key_str = key.decode()
                if key_str.startswith("paths_length_2:"):
                    keys_to_delete.append(key)
                else:
                    break
            for key in keys_to_delete:
                self.db.delete(key)

    def store_concept_pairs_path_length_3(self, concept1, concept2, related):
        key_str = json.dumps([concept1, concept2])
        value_str = json.dumps(related)
        with self.lock:
            self.db.put(self.paths_length_3 + key_str.encode(), value_str.encode())

    def get_store_concept_pairs_path_length_3(self, concept1, concept2):
        key_str = json.dumps([concept1, concept2])
        value = self.db.get(self.paths_length_3 + key_str.encode())
        if value:
            return json.loads(value.decode())
        return None

    def cache_embedding_with_batch_size(self, list_of_pairs, batchsize):
        logging.info('Number of concept pairs to obtain RelBERT embedding = %s', str(len(list_of_pairs)))
        start_time = time.time()
        counter = 0
        working_list = []
        for [a, b] in list_of_pairs:
            if self.check_stored_relbert_embds([a, b]):
                pass
            else:
                working_list.append([a, b])
                if len(working_list) > batchsize:
                    embds = self.model.get_embedding(working_list)
                    for i in range(0, len(working_list)):
                        [c1, c2] = working_list[i]
                        self.set_stored_relbert_embds([c1, c2], embds[i])
                    working_list = []
            counter += 1
            if counter % batchsize == 0:
                logging.info('Number of concept pairs processed = %s, in %s seconds',
                             str(counter), time.time() - start_time)
                start_time = time.time()
        if len(working_list) > 0:
            embds = self.model.get_embedding(working_list)
            for i in range(0, len(working_list)):
                [c1, c2] = working_list[i]
                self.set_stored_relbert_embds([c1, c2], embds[i])
            working_list = []

    def get_embedding_for_one_pair(self, pair):
        embedding = self.get_stored_relbert_embds(pair)
        if embedding is not None:
            return embedding
        else:
            embedding = self.model.get_embedding(pair)
            self.set_stored_relbert_embds(pair, embedding)
            return embedding

    def get_embedding(self, pairs):
        if type(pairs[0]) is list:
            emdedding = self.get_concept_pair_embedding(pairs)
        else:
            concept1, concept2 = pairs
            emdedding = self.get_concept_pair_embedding([[concept1, concept2]])[0]
        return emdedding

    def get_concept_pair_embedding(self, concept_pairs):
        return self.get_stored_embeddings(concept_pairs)

    def get_stored_embeddings1(self, concept1, concept2):
        return self.get_stored_relbert_embds([concept1, concept2])

    def get_stored_embeddings(self, concept_pairs):
        concept_pairs_embd = dict()
        not_in_stored = []
        for [concept1, concept2] in concept_pairs:
            concept_pairs_embd[(concept1, concept2)] = None
            embd = self.get_stored_embeddings1(concept1, concept2)
            if embd is None:
                not_in_stored.append([concept1, concept2])
            else:
                concept_pairs_embd[(concept1, concept2)] = embd
        if not not_in_stored:
            pass
        else:
            embds = self.model.get_embedding(not_in_stored)
            for i in range(0, len(not_in_stored)):
                (concept1, concept2) = not_in_stored[i]
                concept_pairs_embd[(concept1, concept2)] = embds[i]
                self.set_stored_relbert_embds([concept1, concept2], embds[i])
        result = []
        for [concept1, concept2] in concept_pairs:
            result.append(concept_pairs_embd[(concept1, concept2)])
        return result


if __name__ == '__main__':
    initialization()
    cache = Cache(read_only=False, db_path=configuration.rocksdb_path_eval, lock_path=configuration.rocksdb_eval_lock)
    cache.delete_all_stored_related_concepts()
