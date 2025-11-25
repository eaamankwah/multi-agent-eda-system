import json
class SharedMemory:
    def __init__(self):
        self.store_map = {}

    def store(self, key, value):
        self.store_map[key] = value

    def retrieve(self, key):
        return self.store_map.get(key)

    def dump_all(self):
        return self.store_map
