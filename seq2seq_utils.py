import pandas as pd
import codecs
import os

REGEX_PAUSES = ' # | _ | %% %% '


def predict_seq2seq_win(phrase_ser, model):
    """
    Executes seq2seq model from model_dir for phrases in phrase_ser Series. Returnes same-indexed transcription Series.
    """
    test_wdf = sent2word_df(phrase_ser)
    pred = predict_seq2seq(test_wdf.word_win, model)
    test_wdf['pred'] = pred
    pred_ser = test_wdf.groupby('phrase_idx').agg({'pred': merge_trans})
    return pred_ser


def predict_seq2seq(phrase_ser, model):
    write_seq2seq_data('tmp',phrase_ser)
    fpred = 'tmp.phrase.pred'
    os.system('rm %s' % fpred)
    os.system("./infer.sh tmp.phrase %s" % model)
    if not os.path.exists(fpred):
        raise FileNotFoundError('File %s was not created by seq2seq prediction script. Look at infer.sh.log for details.' % fpred)
    with codecs.open(fpred, 'r', encoding='utf-8') as inp:
        lines = [l.strip() for l in inp]
    return pd.Series(lines, phrase_ser.index)


def write_seq2seq_data(fname, phrase_ser, trans_ser=None):
    """
    Write phrases and transcriptions Series phrase_ser and trans_ser to files <fname>.phrase, <fname>.trans
    in seq2seq-compatible format.
    """
    phrase_ser.str.replace(' ','*').apply(lambda x: ' '.join(x)).to_csv(fname + '.phrase', index=False)
    if trans_ser is not None:
        trans_ser.to_csv(fname + '.trans', index=False)


def merge_trans(ser):
    split = ser.str.split(REGEX_PAUSES)
    pauses = ser.str.findall(REGEX_PAUSES)
    wl = []
    for l,p in zip(split,pauses):
        wl.append(l[1] if len(l) > 1 else l[0])
        wl.append(p[1] if len(p) > 1 else p[0] if len(p) > 0 else ' # ')
        if len(l) < 3 or len(p) < 2:
            print('WARNING: less than 3 fields: ', l, p)
    return ''.join(wl[:-1]) # remove last pause (which is before $)


def sent2word_df(phrase_ser, trans_ser=None):
    phrase_l = ('^ ' + phrase_ser + ' $').str.split(' ')

    if trans_ser is not None:
        assert len(phrase_ser) == len(trans_ser)
        trans = ('^ # ' + trans_ser + ' # $')
        trans_l = trans.str.split(REGEX_PAUSES)
        pauses = trans.str.extractall('(' + REGEX_PAUSES + ')')
        mask = phrase_l.map(lambda x: len(x)) == trans_l.map(lambda x: len(x))
        print('%d/%d (%f) have same phrase and transcription word len' % (mask.sum(), len(mask), mask.mean()))
        phrase_l, trans_l = phrase_l[mask], trans_l[mask]

    # split into words
    nwin = [(phrase_idx, l[i - 1:i + 2]) for phrase_idx, l in phrase_l.iteritems() for i in range(1, len(l) - 1)]
    phrase_inds, wins = zip(*nwin)
    phrase_inds, wins = list(phrase_inds), list(wins)
    words = [win[1] for win in wins]

    if trans_ser is not None:
        twins = [l[i - 1:i + 2] for l in trans_l for i in range(1, len(l) - 1)]
        words_trans = [win[1] for win in twins]
        trans_win = [''.join([l[j] + pauses.ix[idx].ix[j].values[0] for j in range(i - 1, i + 1)] + [l[i + 1]])
                     for idx, l in trans_l.iterkv() for i in range(1, len(l) - 1)]
    else:
        words_trans = ['' for w in words]
        trans_win = words_trans

    assert len(words) == len(words_trans)
    assert len(words) == len(trans_win)

    wdf = pd.DataFrame({'phrase_idx': phrase_inds, 'word': words, 'trans': words_trans,
                        'prev_word': [win[0] for win in wins], 'next_word': [win[2] for win in wins],
                        'word_win': [' '.join(win) for win in wins], 'trans_win': trans_win
                        })

    print(len(phrase_ser), 'examples of phrase transcriptions')
    print(len(wdf), 'examples of word transcriptions')
    print(wdf.word.nunique(), 'unique words')
    return wdf