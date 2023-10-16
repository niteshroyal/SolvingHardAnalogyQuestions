import leveldb
import numpy as np
import filelock
from pythonProject.research.reasoning_with_vectors.conf import configuration
from cache import Cache


class LevelDBCache(Cache):
    def __init__(self, read_only=False, db_path=configuration.leveldb_path, lock_path=configuration.leveldb_lock):
        super().__init__(read_only, db_path, lock_path)
        self.db = leveldb.LevelDB(db_path)
        self.lock = filelock.FileLock(lock_path)

    def set_stored_relbert_embds(self, key, value):
        key_str = np.array(key).tobytes()
        value_str = np.array(value).tobytes()
        with self.lock:
            self.db.Put(self.relbert_embeddings + key_str, value_str)

    def get_stored_relbert_embds(self, key):
        key_str = np.array(key).tobytes()
        try:
            value = self.db.Get(self.relbert_embeddings + key_str)
            if value:
                return np.frombuffer(value).tolist()
        except KeyError:
            return None

    def check_stored_relbert_embds(self, key):
        key_str = np.array(key).tobytes()
        try:
            value = self.db.Get(self.relbert_embeddings + key_str)
            return value is not None
        except KeyError:
            return False

    def erase_relbert_embeddings(self, keys):
        for key in keys:
            key_str = np.array(key).tobytes()
            self.db.Delete(self.relbert_embeddings + key_str)