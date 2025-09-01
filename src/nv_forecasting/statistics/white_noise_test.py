from statsmodels.stats.diagnostic import acorr_ljungbox

def white_noise_test(x, alpha=0.05, **kwargs):
    lb = acorr_ljungbox(x, **kwargs)
    lb = lb[lb.lb_pvalue < alpha]
    indices = lb.index

    if len(indices) > 0:
        print('The following lags seem to be significant:')
        for index in indices:
            print(f'{index}, ', end='')
        print('\n')
    else:
        print('No lags seem to be significant.')