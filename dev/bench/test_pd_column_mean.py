

def test_column_mean():
    import timerit
    import pandas as pd
    import numpy as np
    keys = list(map(chr, range(ord('a'), ord('z'))))
    column_data = {
        key: np.random.rand(1000)
        for key in keys
    }
    data_frame = pd.DataFrame(column_data)

    for i in range(3):
        print('\n\n')
        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (DICT[k].mean())', unit='us'):
            with timer:
                result1 = {k: column_data[k].mean() for k in column_data.keys()}

        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (PANDAS[k].mean())', unit='us'):
            with timer:
                result2 = {k: data_frame[k].mean() for k in column_data.keys()}

        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (PANDAS.apply)', unit='us'):
            with timer:
                result3 = data_frame.apply(lambda col: col.mean(), axis=0)
        result3 = result3.to_dict()

        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (PANDAS.mean(axis=0))', unit='us'):
            with timer:
                result4 = data_frame.mean(axis=0)
        result4 = result4.to_dict()

        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (PANDAS.values.mean(axis=0))', unit='us'):
            with timer:
                result5 = data_frame.values.mean(axis=0)
        result5 = dict(zip(keys, result5))

        nkeys = len(keys)
        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (PANDAS.values.T[i].mean())', unit='us'):
            with timer:
                result6 = [data_frame.values.T[i].mean() for i in range(nkeys)]
        result6 = dict(zip(keys, result6))

        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (values[:, i].mean())', unit='us'):
            values = data_frame.values
            with timer:
                result7 = [values[:, i].mean() for i in range(nkeys)]
        result7 = dict(zip(keys, result7))

        npblock = np.hstack([column_data[k][:, None] for k in keys])
        for timer in timerit.Timerit(100, bestof=10, label='Column Mean (npblock[:, i].mean())', unit='us'):
            with timer:
                result8 = [npblock[:, i].mean() for i in range(nkeys)]
        result8 = dict(zip(keys, result8))

        for timer in timerit.Timerit(100, bestof=10, label='Access Values', unit='us'):
            with timer:
                values = data_frame.values

        for timer in timerit.Timerit(100, bestof=10, label='pass', unit='us'):
            with timer:
                pass

    assert result1 == result2
    assert result2 == result3
    assert all(np.isclose(result3[k], result4[k]) for k in keys)
    assert all(np.isclose(result4[k], result5[k]) for k in keys)
    assert all(np.isclose(result5[k], result6[k]) for k in keys)

    assert all(np.isclose(result6[k], result7[k]) for k in keys)
    assert all(np.isclose(result7[k], result8[k]) for k in keys)
