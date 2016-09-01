import numpy as np
from asq.initiators import query as asq_query
import pandas as pd
__author__ = 'ORI'



if __name__ == "__main__":
    feb_6_res = np.load(
        r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\ipytho_notebook\reports\6_feb_2016_explore_LSTM.npy')
    all_results = []
    for elm in feb_6_res:
        name = elm['subject_name']
        accuracies_per_subject = []
        accuracies_per_subject = []
        for result in elm['subject_results'].values():
            # average the results of the different trial and compute their standard deviation
            accuracies_per_subject.append(result['acc_by_rep'])
        max_res = np.array(accuracies_per_subject).max()
        average_acc_per_subject = np.array(accuracies_per_subject).mean()
        std_acc_per_subject = np.array(accuracies_per_subject).std()
        all_results.append([name,average_acc_per_subject, std_acc_per_subject,max_res])

    df = pd.DataFrame(data=all_results, columns=['name','average_acc','std','max'])
    df.to_excel(r"C:\git\thesis_clean_v2\res1.xls")

    pass
