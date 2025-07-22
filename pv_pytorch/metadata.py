

import json


class LabelMapping:
    def __init__(self):
        self.mapping   = {}
        self.negatives = [] #list of keys that would indicate negative results

    def __len__(self):
        return len(set([self.mapping[key] for key in self.mapping]))

    def __str__(self):
        return str([key for key in self.mapping])

    def __repr__(self):
        return repr(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __iter__(self):
        for key in self.mapping:
            yield key

    def _get_labels():
        return set([self.mapping[key] for key in self.mapping])

    def load(self, file):
        with open(file, "r") as f:
            data = json.load(f)

            for keys_to_label in data["mapping"]:
                keys  = keys_to_label["keys"]
                label = keys_to_label["label"]

                for key in keys:
                    self.mapping[key] = label

            self.negatives = data["negatives"]

    def save(self, file):
        mapping_set = {}
        for key in self.mapping:

            if self.mapping[key] in mapping_set:
                mapping_set[self.mapping[key]].append(key) #append key to list mapped by label
            else:
                mapping_set[self.mapping[key]] = [key]


        mapping_list = [{"keys" : mapping_set[label], "label" : label} for label in mapping_set]

        with open(file, "w") as f:
            json.dump({
                "mapping" : mapping_list,
                "negatives" : self.negatives
                }, f, indent=4)
