from scipy import stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import pingouin as pg


def bin1_vs_bin5_stats(rewp_per_subject, conditions):
    results = {}
    pvals = []
    
    for cond in conditions:
        bin1 = rewp_per_subject[cond][:, 0]
        bin5 = rewp_per_subject[cond][:, -1]
        mask = ~np.isnan(bin1) & ~np.isnan(bin5)
        bin1_clean, bin5_clean = bin1[mask], bin5[mask]
        
        t, p = stats.ttest_rel(bin1_clean, bin5_clean)
        diff = bin1_clean - bin5_clean
        d = diff.mean() / diff.std()
        pvals.append(p)
        
        results[cond] = {
            't': t, 'p': p, 'd': d,
            'bin1_mean': bin1_clean.mean(),
            'bin5_mean': bin5_clean.mean()
        }

    reject, p_corrected, _, _ = multipletests(pvals, method='bonferroni')
    for cond, p_corr, rej in zip(conditions, p_corrected, reject):
        results[cond]['p_corrected'] = p_corr
        results[cond]['reject_h0'] = rej

    return results


def rm_anova_stats(rewp_per_subject, conditions, n_bins, subjects):
    records = []
    for s_idx, subj in enumerate(subjects):
        for cond in conditions:
            for b in range(n_bins):
                records.append({
                    'subject': subj,
                    'condition': cond,
                    'bin': b + 1,
                    'rewp': rewp_per_subject[cond][s_idx, b]
                })

    df_balanced = pd.DataFrame(records)
    subjects_with_nan = df_balanced[df_balanced['rewp'].isna()]['subject'].unique()
    df_balanced = df_balanced[~df_balanced['subject'].isin(subjects_with_nan)]

    aov = pg.rm_anova(
        data=df_balanced,
        dv='rewp',
        within=['condition', 'bin'],
        subject='subject',
        effsize='np2',
        correction=True
    )
    
    return aov[['Source', 'F', 'ddof1', 'ddof2', 'p-unc', 'p-GG-corr', 'np2', 'eps']]