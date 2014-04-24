ROOT_LABEL	= 'ROOT'
NUM_THREADS	= 1

-- dpiornn settings
IORNN_CONTEXT_SIZE	= 3

-- depstruct settings
DEPSTRUCT_N_DEPS = 200

-- for training
TRAIN_LAMBDA		= 1e-4
TRAIN_LAMBDA_L		= 1e-10
TRAIN_BATCHSIZE		= 16
TRAIN_MAX_N_EPOCHS	= 100
TRAIN_UPDATE_L		= true

TRAIN_WEIGHT_LEARNING_RATE	= 0.01
TRAIN_VOCA_LEARNING_RATE	= 0.01

-- for eval
EVAL_TOOL_PATH	= '../tools/eval-dep.pl'
EVAL_EMAIL_ADDR	= 'lephong.xyz@gmail.com'
