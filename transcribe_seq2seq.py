# coding=utf-8

import codecs
import argparse
import pandas as pd

from seq2seq_utils import predict_seq2seq_win
from stress_utils import add_stress, load_stress_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fin", default="test_in1.txt", help="path to the file with phrases to transcribe")
    parser.add_argument("-fout", default="test_out1.txt", help="path to the file to write transcriptions to")
    parser.add_argument("-model", default="word3stress_bahdanau/model.ckpt-14850", help="directory with saved seq2seq model")

    args = parser.parse_args()
    fin = args.fin
    fout = args.fout
    model = args.model


    # Load input phrases to DataFrame
    print('Loading input phrases from %s ...' % fin)
    with codecs.open(fin, 'r', encoding='utf-8') as inp:
        lines = [x.strip() for x in inp]
    pdf = pd.DataFrame(lines, columns=['phrase'])

    # Convert to lower, replace all non-letters with spaces, remove repeating spaces
    # finally only letters from train set are left
    pdf.phrase = pdf.phrase.str.lower().str.replace('[^а-яё ]', ' ').str.replace(' +', ' ').str.strip()

    print('Loading stress dictionary ...')
    stress_dict = load_stress_dict()
    print('Adding stress to phrases ...')
    pdf['phrase'] = add_stress(pdf.phrase, stress_dict)

    print('Executing seq2seq model from %s ...' % model)
    pred = predict_seq2seq_win(pdf.phrase, model)

    print('Writing transcriptions to %s ...' % fout)
    pred.to_csv(fout, index=False, header=None)


if __name__ == "__main__":
    main()