import collections
import glob
import random
import pandas as pd

import json_lines


def extract_user_data(data, user, label):
    posts = random.sample(user['posts'], 12)
    texts = [t['text'] for t in posts if t['text'] != ""]
    if texts is None or len(texts) == 0:
        print("text is empty or null! for id: ", user['id'])
        return
    text_str = ' '.join(texts)
    text_str = text_str[:512] + (text_str[512:] and '')
    data.setdefault('sentence', []).append(text_str)
    data.setdefault('id', []).append(user['id'])
    data.setdefault('label', []).append(label)


class Processor:
    def __init__(self, mental_limit, controls_limit):

        self.controls_limit = controls_limit
        self.mental_limit = mental_limit
        self.controls_amount = 0
        self.mental_amount = 0
        self.should_stop = False

    def read_files(self, files):
        frames = []
        for f in files:
            if not self.should_stop:
                frame = self.__read_file(f)
                frames.append(frame)

        return pd.concat(frames)

    def __read_file(self, file):
        data = collections.defaultdict(list)
        with open(file, 'rb') as f:
            try:
                for user in json_lines.reader(f, broken=True):
                    if 'control' in user['label'] and self.controls_amount < self.controls_limit:
                        extract_user_data(data, user, 0)
                        self.controls_amount += 1
                    elif 'depression' in user['label'] and self.mental_amount < self.mental_limit:
                        extract_user_data(data, user, 1)
                        self.mental_amount += 1
                    if self.mental_amount == self.mental_limit and self.controls_amount == self.controls_limit:
                        self.should_stop = True
                        break
            except RuntimeError as err:
                print("Something is wrong with the left flange:".format(err))

        return pd.DataFrame(data)


if __name__ == '__main__':
    data = [("train", 1500, 1500), ("test", 700, 700), ("dev", 150, 150)]
    for d in data:
        processor = Processor(d[1], d[2])
        files = glob.glob(f'../resources/{d[0]}/*.jl')
        data = processor.read_files(files)
        data.loc[-1] = data.dtypes
        data.index = data.index + 1
        data.sort_index(inplace=True)
        # Then save it to a csv
        data.to_csv(f'../resources/{d[0]}_data_small.csv', index=False)

