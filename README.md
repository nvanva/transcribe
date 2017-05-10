# transcribe

## Requirements
  * Python3
  * Tensorflow version 1.1
  * Seq2seq: https://google.github.io/seq2seq/getting_started/
  * Pandas
  
## Usage

To transcribe phrases in test_in1.txt using default model run
```bash
python3 transcribe_seq2seq.py
```
Transcriptions will be written to test_out1.txt.

Input, output files and model can be specified explicitly:
```bash
python3 transcribe_seq2seq.py -fin test_in1.txt -fout test_out1.txt -model word3stress_bahdanau/model.ckpt-14850
```
