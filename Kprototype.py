

import datatable as dt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer

class KPrototypesTransformer(CustomUnsupervisedTransformer):
    @staticmethod
    def get_default_properties():
        _modules_needed_by_name = ["kmodes"]

        return dict(col_type="mixed", min_cols=2, max_cols="all")  # Adjust for mixed types

    @staticmethod
    def get_parameter_choices():
        return dict(n_clusters=[2, 3, 4, 5, 6], init=['Huang', 'Cao', 'random'])

    def __init__(self, n_clusters=None, init='Huang', **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.init = init

    def fit_transform(self, X: dt.Frame, y: np.array = None):

        # KPrototypes needs to know which columns are categorical. Assuming last column is categorical here as an example.
        categorical_indices = [X.shape[1] - 1]  # Example: Assuming last column is categorical
        
        X_pd = X.to_pandas().fillna(0)  # Convert to pandas DataFrame and fill NA values
        self.model = KPrototypes(n_clusters=self.n_clusters, init=self.init, verbose=1)
        clusters = self.model.fit_predict(X_pd, categorical=categorical_indices)
        
        return clusters

    def transform(self, X: dt.Frame, y: np.array = None):
        return self.fit_transform(X)


class KPrototypesModel(CustomUnsupervisedModel):
    _included_pretransformers = ['StdFreqPreTransformer']  # Might need adjustment based on data preprocessing needs

    _included_transformers = ["KPrototypesTransformer"]

    _included_scorers = ['SilhouetteScorer', 'CalinskiHarabaszScorer', 'DaviesBouldinScorer']  # Keep or adjust scorers as needed
