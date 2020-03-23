import pandas as pd
from functools import reduce
import os
import string
from ast import literal_eval
from rake_nltk import Rake
import operator


class FeaturesExtractor:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.rake = Rake()

        self.cols = ['id', 'title', 'popularity',
                     'overview', 'tagline',
                     'keywords', 'cast', 'genres']

        self.movies_metadata = 'movies_metadata.csv'
        self.movies_metadata_clean = 'movies_metadata_clean.csv'
        self.movies_data_files = ['keywords.csv', 'credits.csv',
                                  self.movies_metadata_clean]

    def run(self, unique=False):
        if not os.path.exists(os.path.join(self.dataset_path,
                                           self.movies_metadata_clean)):
            self.fix_file()
        data = self.load_data()
        self.process_keywords_columns(data)
        self.process_text_columns(data)
        self.add_combined_features(data, unique)
        return data

    def fix_file(self):
        df = pd.read_csv(os.path.join(self.dataset_path,
                                      self.movies_metadata))
        df = df[df['id'].str.isdigit()]
        df['id'] = df['id'].astype(int)
        df.to_csv(os.path.join(self.dataset_path, self.movies_metadata_clean),
                  index=False)

    def load_data(self, small=True):
        dfs = [pd.read_csv(os.path.join(self.dataset_path, f_name))
               for f_name in self.movies_data_files]
        data = reduce(lambda df1, df2: pd.merge(df1, df2, on='id', how='outer'),
                      dfs)
        data = data[self.cols].groupby('id').first().reset_index(level=0)

        if small:
            data = self.reduce_num_of_movies(data)

        return data.reset_index(drop=True)

    def reduce_num_of_movies(self, data):
        ratings_small = pd.read_csv(
            os.path.join(self.dataset_path, 'ratings_small.csv'))
        movies_id_small = set(ratings_small['movieId'])
        return data[data['id'].isin(movies_id_small)]

    def process_keywords_columns(self, data):
        data['keywords'] = data['keywords'].apply(
            lambda value: self.process_dict(value, self.clean_keywords))
        data['cast'] = data['cast'].apply(
            lambda value: self.process_dict(value, self.clean_elements))
        data['genres'] = data['genres'].apply(
            lambda value: self.process_dict(value, self.clean_elements))

    def process_text_columns(self, data):
        data['overview'] = self.process_text_column(data['overview'], 7)
        data['tagline'] = self.process_text_column(data['tagline'], 3)

    def add_combined_features(self, data, unique):
        def get_words(row):
            words = row.values.astype(str)
            return set(words) if unique else words

        data['combined'] = data[self.cols[3:]].apply(
            lambda row: ' '.join(get_words(row)).strip(' '), axis=1)

    def process_dict(self, value, clean):
        return clean(self.extract_name(literal_eval(value)))

    def process_text_column(self, col, num_keywords):
        return col.fillna('').astype(str).apply(
            lambda text: self.extract_keywords(text, num_keywords))

    def clean_keywords(self, values):
        return ' '.join(set([word for value in values for word in
                             self.remove_punctation(value).lower().split()]))

    def clean_elements(self, values):
        return ' '.join(
            [self.remove_punctation(value).lower().replace(' ', '') for value in
             values])

    def extract_keywords(self, text, n):
        text = self.remove_punctation(text).lower()
        self.rake.extract_keywords_from_text(text)
        candidates = self.rake.get_word_degrees()
        sorted_keywords = sorted(candidates.items(), key=operator.itemgetter(1),
                                 reverse=True)
        return ' '.join([word for word, score in sorted_keywords[:n]])

    def extract_name(self, tags):
        return [tag['name'] for tag in tags]

    def remove_punctation(self, value):
        return value.translate(str.maketrans('', '', string.punctuation))




