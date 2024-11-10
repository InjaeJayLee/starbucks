import pandas as pd
import numpy as np
import math
import json
import re


class Preprocessing:
    def __init__(self,
                 portfolio: pd.DataFrame,
                 profile: pd.DataFrame,
                 transcript: pd.DataFrame):
        """preprocess transcript by combining it with portfolio and profile data"""

        self.portfolio = portfolio
        self.profile = profile
        self.transcript = transcript

    def run(self) -> pd.DataFrame:
        self._impute_profile_data()
        self._clean_dict_keys()
        self._add_offer_details()
        self._take_completion_reward_out()
        self._sort()
        return self._combine_transcript_with_profile()

    def _impute_profile_data(self):
        self.profile['gender'].fillna('O', inplace=True)
        # it is highly likely that users with this age didn't agree to providing information of their age
        self.profile.loc[self.profile['age'] == 118, 'age'] = np.nan

    def _clean_dict_keys(self):
        def remove_white_space_in_value_key(value: dict):
            wrong_keys = [key for key in value.keys() if re.search(r'\s', key)]
            for key in wrong_keys:
                val = value.pop(key)
                new_key = key.replace(" ", "_")
                value[new_key] = val
        self.transcript['value'].apply(remove_white_space_in_value_key)

    def _add_offer_details(self):
        def get_offer_detail(offer_id: str, info_type: str):
            return self.portfolio.loc[self.portfolio['id'] == offer_id, info_type].values[0]

        self.transcript['offer_id'] = np.nan
        self.transcript['offer_reward'] = np.nan
        self.transcript['offer_channels'] = np.nan
        self.transcript['offer_difficulty'] = np.nan
        self.transcript['offer_duration'] = np.nan
        self.transcript['offer_type'] = np.nan

        offer_flags = self.transcript['event'].str.contains('offer')

        self.transcript.loc[offer_flags, 'offer_id'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: x['offer_id'])
        self.transcript.loc[offer_flags, 'offer_reward'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: get_offer_detail(x['offer_id'], 'reward'))
        self.transcript.loc[offer_flags, 'offer_channels'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: get_offer_detail(x['offer_id'], 'channels'))
        self.transcript.loc[offer_flags, 'offer_difficulty'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: get_offer_detail(x['offer_id'], 'difficulty'))
        self.transcript.loc[offer_flags, 'offer_duration'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: 24 * get_offer_detail(x['offer_id'], 'duration'))
        self.transcript.loc[offer_flags, 'offer_type'] = self.transcript.loc[offer_flags, 'value'].apply(
            lambda x: get_offer_detail(x['offer_id'], 'offer_type'))

    def _take_completion_reward_out(self):
        self.transcript['actual_reward'] = 0
        completion_flags = self.transcript['event'] == 'offer completed'
        self.transcript.loc[completion_flags, 'actual_reward'] = self.transcript.loc[completion_flags, 'value'].apply(
            lambda x: x['reward'])

    def _sort(self):
        self.transcript.sort_values(['person', 'time'], ascending=[True, True], inplace=True)
        self.transcript.reset_index(drop=True, inplace=True)

    def _combine_transcript_with_profile(self):
        return (self.transcript.merge(self.profile,left_on='person',right_on='id',how='left')
                .drop(columns=['id', 'value'])
                .rename(columns={'person': 'user_id',
                                 'age': 'user_age',
                                 'gender': 'user_gender',
                                 'income': 'user_income',
                                 'became_member_on': 'user_became_member_on'}))
