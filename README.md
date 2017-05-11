# transcribe

## Requirements
 * Python3
 * Tensorflow version 1.1
 * Seq2seq: https://google.github.io/seq2seq/getting_started/
 * Pandas

If you use Anaconda, all dependencies can be installed into separate Anaconda environment:
  ```bash
  conda create -n python36_tf11 python=3.6
  source activate python36_tf11
  export Q=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl
  pip install --ignore-installed --upgrade $Q
  conda install pandas
  
  git clone https://github.com/google/seq2seq.git
  cd seq2seq
  pip install -e .
  ```


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
