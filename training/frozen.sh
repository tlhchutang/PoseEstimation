#python3 src/gen_frozen_pb.py --checkpoint=/raid/tangc/mv2_cpm/models/mv2_cpm_batch-128_lr-0.001_gpus-1_192x192_experiments-mv2_cpm/model-420000 --output_graph=./frozen_pb/frozen_model.pb

python3 src/gen_frozen_pb.py --checkpoint=/raid/tangc/mv2_cpm_augmentation/models/mv2_cpm_batch-128_lr-0.001_gpus-1_192x192_experiments-mv2_cpm_augmentation/model-200000 --output_graph=./frozen_pb/frozen_model.pb


