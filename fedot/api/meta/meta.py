import re
import numpy as np
import pandas as pd
import pickle

import pymfe
from pymfe.mfe import MFE
from pymfe.concept import MFEConcept
from pymfe.complexity import MFEComplexity
from pymfe.general import MFEGeneral
from pymfe.statistical import MFEStatistical
from pymfe.landmarking import MFELandmarking
from pymfe.model_based import MFEModelBased
from pymfe.info_theory import MFEInfoTheory
from pymfe.clustering import MFEClustering

class MetaRecommender:
    def __init__(self):
        self.metabase = pd.read_csv('fedot/api/meta/meta_base_v1.csv')
        self.metabase['idx'] = self.metabase.index
        self.meta_recommender = pickle.load( open( "fedot/api/meta/meta-recommender.p", "rb" ) )

    def parse_configs(self,configs):
        params = dict()
        params['max_depth'] = int(re.findall('[0-9]', configs)[0])
        params['max_arity'] = int(re.findall('[0-9]', configs)[0])
        params['available_operations'] = re.sub('[\,\[\]]', '', re.findall('\[.{0,}\]', configs)[0]).split(' ')
        return params

    def get_metafeats(self,X,y, target_col='target'):
        X,y = X.values, y.values
        metafeatures = np.empty((24, 1))

        metafeatures[0, 0] = np.std(MFEConcept.ft_conceptvar(X, y), ddof=1)
        metafeatures[1, 0] = MFEComplexity.ft_lsc(X, y)
        metafeatures[2, 0] = np.std(MFEGeneral.ft_freq_class(X, y), ddof=1)
        metafeatures[3, 0] = np.mean(MFEComplexity.ft_n3(X, y))
        metafeatures[4, 0] = MFEStatistical.ft_nr_cor_attr(X)
        metafeatures[5, 0] = np.mean(MFEComplexity.ft_f1(X, y))
        metafeatures[6, 0] = MFEComplexity.ft_c2(X, y)
        metafeatures[7, 0] = np.mean(MFEComplexity.ft_f4(X, y))
        metafeatures[8, 0] = MFEComplexity.ft_n1(X, y)
        metafeatures[9, 0] = np.mean(MFEComplexity.ft_l1(X, y))
        metafeatures[10, 0] = np.mean(MFELandmarking.ft_best_node(X, y, score=pymfe.scoring.accuracy))
        metafeatures[11, 0] = np.mean(MFELandmarking.ft_linear_discr(X, y, score=pymfe.scoring.accuracy))
        metafeatures[12, 0] = MFE(groups=["model-based"]).fit(X, y).extract()[1][7]
        metafeatures[13, 0] = MFEGeneral.ft_nr_class(X, y)
        metafeatures[14, 0] = np.mean(MFEGeneral.ft_freq_class(X, y))
        metafeatures[15, 0] = np.mean(MFELandmarking.ft_elite_nn(X, y, score=pymfe.scoring.accuracy))
        metafeatures[16, 0] = np.mean(MFEConcept.ft_conceptvar(X, y))
        metafeatures[17, 0] = np.mean(MFEComplexity.ft_l2(X, y))
        metafeatures[18, 0] = np.mean(MFEComplexity.ft_f1v(X, y))
        metafeatures[19, 0] = MFEClustering.ft_nre(X, y)
        metafeatures[20, 0] = np.mean(MFELandmarking.ft_random_node(X, y, score=pymfe.scoring.accuracy))
        metafeatures[21, 0] = np.mean(MFELandmarking.ft_worst_node(X, y, score=pymfe.scoring.accuracy))
        metafeatures[22, 0] = np.mean(MFEComplexity.ft_l3(X, y))
        metafeatures[23, 0] = np.mean(np.log1p(MFEInfoTheory.ft_class_ent(X, y)))
        return metafeatures.T

    def recommend_params(self, df):
        X,y = df.drop(['target'],1), df['target']
        temp_df_metafeats = self.get_metafeats(X,y)
        temp_meta_idx = self.meta_recommender.predict(pd.DataFrame(temp_df_metafeats).fillna(0)\
                                                      .replace(-np.inf, 0).replace(np.inf, 0))[0]
        temp_params = self.parse_configs(self.metabase.loc[temp_meta_idx, '1'])
        return temp_params
