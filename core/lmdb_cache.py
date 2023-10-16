import os
import lmdb
import json
import time
import redis
import pickle
import rocksdb
import logging
import resource
from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.cache import Cache


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def transfer_rocksdb_to_lmdb(rocks_db_path, lmdb_path, db_name):
    options = rocksdb.Options()
    options.create_if_missing = True
    options.max_open_files = min(300000, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    rocks_db = rocksdb.DB(rocks_db_path, options)
    lmdb_env = lmdb.open(lmdb_path, max_dbs=3, map_size=1073741824 * 200)
    lmdb_db = lmdb_env.open_db(db_name.encode())
    batch_size = 10000
    it = rocks_db.iteritems()
    it.seek(db_name.encode())
    batch = []
    counter = 1
    for key_bytes, value_bytes in it:
        if not key_bytes.startswith(db_name.encode()):
            break
        key = json.loads(key_bytes.decode().split(db_name, 1)[1])
        value = json.loads(value_bytes.decode())
        batch.append((key, value))
        if len(batch) >= batch_size:
            with lmdb_env.begin(write=True, db=lmdb_db) as txn:
                for key, value in batch:
                    txn.put(pickle.dumps(key), pickle.dumps(value))
            batch = []
            logging.info(f'Partial transfer till record no. {counter} of database = {db_name} '
                         f'from rocksdb at {rocks_db_path} to lmdb at {lmdb_path}')
        counter += 1
    if batch:
        with lmdb_env.begin(write=True, db=lmdb_db) as txn:
            for key, value in batch:
                txn.put(pickle.dumps(key), pickle.dumps(value))
        logging.info(f'Complete transfer till record no. {counter} of database = {db_name} '
                     f'from rocksdb at {rocks_db_path} to lmdb at {lmdb_path}')


class LMDB(Cache):
    def __init__(self, read_only=False, db_path=configuration.lmdb_path, lock_path=None):
        self.clear_redis_cache_in_the_start = configuration.clear_redis_cache_in_the_start
        self.env = None
        self.relbert_db = None
        self.paths_2_db = None
        self.paths_3_db = None
        self.redis_cache = None
        super().__init__(read_only, db_path, lock_path)
        self.relbert_embeddings_decoded = self.relbert_embeddings.decode()
        self.paths_length_2_decoded = self.paths_length_2.decode()
        self.paths_length_3_decoded = self.paths_length_3.decode()

    def init_db(self, read_only, db_path, lock_path):
        self.env = lmdb.open(db_path, max_dbs=3, readonly=read_only, map_size=1073741824 * 200)
        self.relbert_db = self.env.open_db(self.relbert_embeddings)
        self.paths_2_db = self.env.open_db(self.paths_length_2)
        self.paths_3_db = self.env.open_db(self.paths_length_3)
        self.redis_cache = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.wait_till_redis_is_available()
        if self.clear_redis_cache_in_the_start:
            self.redis_cache.flushdb()
        else:
            pass
        self.wait_till_redis_is_available()

    def set_stored_relbert_embds(self, key, value):
        key_bytes = pickle.dumps(key)
        value_bytes = pickle.dumps(value)
        with self.env.begin(write=True, db=self.relbert_db) as txn:
            txn.put(key_bytes, value_bytes)
        self.redis_cache.hset(self.relbert_embeddings_decoded, key_bytes, value_bytes)

    def get_stored_relbert_embds(self, key):
        key_bytes = pickle.dumps(key)
        value_bytes = self.redis_cache.hget(self.relbert_embeddings_decoded, key_bytes)
        if value_bytes is None:
            with self.env.begin(db=self.relbert_db) as txn:
                value_bytes = txn.get(key_bytes)
            if value_bytes:
                self.redis_cache.hset(self.relbert_embeddings_decoded, key_bytes, value_bytes)
        if value_bytes:
            return pickle.loads(value_bytes)
        else:
            return None

    def iter_stored_relbert_embds(self):
        with self.env.begin(db=self.relbert_db) as txn:
            cursor = txn.cursor()
            for key_bytes, value_bytes in cursor:
                key = pickle.loads(key_bytes)
                value = pickle.loads(value_bytes)
                yield key, value

    def check_stored_relbert_embds(self, key):
        key_bytes = pickle.dumps(key)
        with self.env.begin(db=self.relbert_db) as txn:
            with txn.cursor() as cursor:
                return cursor.set_key(key_bytes)

    def all_stored_relbert_embds(self):
        with self.env.begin(db=self.relbert_db) as txn:
            cursor = txn.cursor()
            result = []
            for key_bytes, value_bytes in cursor:
                key = pickle.loads(key_bytes)
                result.append(key)
            return result

    def erase_relbert_embeddings(self, keys):
        with self.env.begin(write=True, db=self.relbert_db) as txn:
            for key in keys:
                key_bytes = pickle.dumps(key)
                self.redis_cache.hdel(self.relbert_embeddings_decoded, key_bytes)
                txn.delete(key_bytes)

    def store_related_concepts(self, concept1, concept2, related):
        key_bytes = pickle.dumps([concept1, concept2])
        value_bytes = pickle.dumps(related)
        with self.env.begin(write=True, db=self.paths_2_db) as txn:
            txn.put(key_bytes, value_bytes)
        self.redis_cache.hset(self.paths_length_2_decoded, key_bytes, value_bytes)

    def get_store_related_concepts(self, concept1, concept2):
        key_bytes = pickle.dumps([concept1, concept2])
        value_bytes = self.redis_cache.hget(self.paths_length_2_decoded, key_bytes)
        if value_bytes is None:
            with self.env.begin(db=self.paths_2_db) as txn:
                value_bytes = txn.get(key_bytes)
            if value_bytes:
                self.redis_cache.hset(self.paths_length_2_decoded, key_bytes, value_bytes)
        if value_bytes:
            return pickle.loads(value_bytes)
        else:
            return None

    def delete_all_stored_related_concepts(self):
        keys = []
        with self.env.begin(write=True, db=self.paths_2_db) as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                keys.append(key_bytes)
        with self.env.begin(write=True, db=self.paths_2_db) as txn:
            for key in keys:
                self.redis_cache.hdel(self.paths_length_2_decoded, key)
                txn.delete(key)

    def store_concept_pairs_path_length_3(self, concept1, concept2, related):
        key_bytes = pickle.dumps([concept1, concept2])
        value_bytes = pickle.dumps(related)
        with self.env.begin(write=True, db=self.paths_3_db) as txn:
            txn.put(key_bytes, value_bytes)
        self.redis_cache.hset(self.paths_length_3_decoded, key_bytes, value_bytes)

    def get_store_concept_pairs_path_length_3(self, concept1, concept2):
        key_bytes = pickle.dumps([concept1, concept2])
        value_bytes = self.redis_cache.hget(self.paths_length_3_decoded, key_bytes)
        if value_bytes is None:
            with self.env.begin(db=self.paths_3_db) as txn:
                value_bytes = txn.get(key_bytes)
            if value_bytes:
                self.redis_cache.hset(self.paths_length_3_decoded, key_bytes, value_bytes)
        if value_bytes:
            return pickle.loads(value_bytes)
        else:
            return None

    def wait_till_redis_is_available(self):
        result = False
        while not result:
            try:
                self.redis_cache.ping()
                result = True
            except redis.exceptions.BusyLoadingError:
                time.sleep(10)


if __name__ == '__main__':
    current_soft_limit, current_hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft_limit = min(8192, current_hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, current_hard_limit))
    initialization()
    transfer_rocksdb_to_lmdb(configuration.rocksdb_path_eval, configuration.lmdb_path, 'relbert_embeddings:')
    transfer_rocksdb_to_lmdb(configuration.rocksdb_path_eval, configuration.lmdb_path, 'paths_length_2:')
    transfer_rocksdb_to_lmdb(configuration.rocksdb_path_eval, configuration.lmdb_path, 'paths_length_3:')
