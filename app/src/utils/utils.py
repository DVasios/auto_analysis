# Utils
def has_missing_data(df_descr):
    for f, d in df_descr['features'].items():

        if (d['eda']['missing_data']['percentage'] > 0.0):
            print('yes')
            print(f)