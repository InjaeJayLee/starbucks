import pandas as pd


def evaluate_offers(transcript: pd.DataFrame):
    """
    :param transcript: a preprocessed transcript after using Preprocessing class
    :return: a DataFrame with rows of received offer events with the result about how the offer worked
    ('offer_result' column)
    """

    counts = transcript.groupby(['user_id', 'offer_id'])['event'].value_counts().unstack()
    counts['offer_result'] = ''

    def evaluate_how_offer_worked(row):
        if row['offer received'] >= 1 and row['offer viewed'] >= 1 and row['offer completed'] >= 1:
            return 'offer worked'
        elif row['offer received'] >= 1 and row['offer completed'] >= 1:
            return 'purchased without offer'
        else:
            return 'offer did not work'

    counts['offer_result'] = counts.apply(evaluate_how_offer_worked, axis=1)
    offer_results = counts['offer_result'].reset_index()

    transcript_received = transcript.loc[transcript['event'] == 'offer received'].reset_index(drop=True)

    return pd.merge(transcript_received, offer_results, on=['user_id', 'offer_id'], how='left')


def create_dataset(offer_results: pd.DataFrame):
    offer_results['target'] = offer_results['offer_result'].apply(lambda x: 1 if x == 'offer worked' else 0)

    columns = offer_results.columns
    reorder_cols = ['target']
    reorder_cols.extend([c for c in columns if c not in reorder_cols])

    offer_results['channel_email'] = 0
    offer_results['channel_mobile'] = 0
    offer_results['channel_social'] = 0
    offer_results['channel_web'] = 0
    offer_results['channel_email'] = offer_results['offer_channels'].apply(lambda x: int('email' in x))
    offer_results['channel_mobile'] = offer_results['offer_channels'].apply(lambda x: int('mobile' in x))
    offer_results['channel_social'] = offer_results['offer_channels'].apply(lambda x: int('social' in x))
    offer_results['channel_web'] = offer_results['offer_channels'].apply(lambda x: int('web' in x))

    offer_results.drop(columns=['event', 'user_id', 'offer_id', 'actual_reward', 'offer_channels',
                                'user_became_member_on', 'offer_result', 'time'],
                       inplace=True)

    return pd.get_dummies(offer_results, drop_first=True, columns=['user_gender', 'offer_type'])
