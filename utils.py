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

    results = pd.merge(transcript_received, offer_results, on=['user_id', 'offer_id'], how='left')
    columns = results.columns
    reorder_cols = ['user_id', 'offer_id', 'offer_result']
    reorder_cols.extend([c for c in columns if c not in reorder_cols])
    results = results[reorder_cols].drop(columns=['event'])
    return results
