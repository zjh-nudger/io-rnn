--torch.setdefaulttensortype('torch.FloatTensor')

N_THREADS = 4
SUBBATCH_SIZE = 10

-- ********** filenames ********* --
TRAIN_FILENAME = 'train'
DEV_FILENAME = 'dev.conll'
KBEST_DEV_FILENAME = 'dev.conll'

WORD_FILENAME = 'words.lst'
WCODE_FILENAME = 'word_code.lst'
POS_FILENAME = 'pos.lst'
DEPREL_FILENAME = 'deprel.lst'

-- ********** params ********** --
ROOT_LABEL	= 'ROOT'

-- capital feature
ALL_LOWER = 1
N_CAP_FEAT = 1

-- direction
DIR_L 	= 1
DIR_R 	= 2

-- depstruct settings
DEPSTRUCT_N_DEPS = 50

-- for training
MIN_OCCURS_THRESHOLD	= 3
TRAIN_LAMBDA		= 1e-4
TRAIN_LAMBDA_L		= 1e-10
TRAIN_BATCHSIZE		= 32
TRAIN_MAX_N_EPOCHS	= 500
TRAIN_UPDATE_L		= true

TRAIN_WEIGHT_LEARNING_RATE	= 0.1
TRAIN_VOCA_LEARNING_RATE	= 0.1

-- for eval
EVAL_TOOL_PATH	= '../tools/eval-dep.pl'
--EVAL_EMAIL_ADDR	= 'lephong.xyz@gmail.com'

K		= 1
alpha	= 0

K_range	= nil -- to choose K [for development], set nil when testing
alpha_range = nil -- to choose alpha (alpha * mstscore + (1-alpha) * iornnscore) [for development] set nil when testing
punc = true -- taking punc into account for evaluation 

