
PHRASES=$1
CHECKPOINT=$2
MODEL_DIR=$(dirname $CHECKPOINT)
echo $MODEL_DIR

python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --checkpoint_path $CHECKPOINT \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $PHRASES" \
  >  ${PHRASES}.pred 2> infer.sh.log
