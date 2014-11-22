torch.setdefaulttensortype('torch.FloatTensor')

-- ********** filenames ********* --
TRAIN_FILENAME = 'train.conll.mst'
DEV_FILENAME = 'dev.conll'
KBEST_DEV_FILENAME = 'dev-50-best-mst2ndorder.conll'

WORD_FILENAME = 'words.lst'
WEMB_FILENAME = 'embeddings.txt'
WCODE_FILENAME = 'word_code.lst'
POS_FILENAME = 'pos.lst'
DEPREL_FILENAME = 'deprel.lst'

-- ********** params ********** --

ROOT_LABEL	= 'ROOT'
NUM_THREADS	= 1

N_PREV_TREES = 0

-- capital feature
ALL_LOWER = 1
ALL_UPPER = 2
FIRST_UPPER = 3
NOT_FIRST_UPPER = 4
N_CAP_FEAT = 4

-- direction
DIR_L 	= 1
DIR_R 	= 2

-- distance feature
DIST_1		= 1
DIST_2		= 2
DIST_3_6	= 3
DIST_7_INF	= 4
N_DIST_FEAT	= 4


-- depstruct settings
DEPSTRUCT_N_DEPS = 80

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
EVAL_EMAIL_ADDR	= 'lephong.xyz@gmail.com'

USE_GOLD_PREV_TREES = false

K		= 30
alpha	= 0

K_range	= {2,30} -- to choose K [for development], set nil when testing
alpha_range = {0,1} -- to choose alpha (alpha * mstscore + (1-alpha) * iornnscore) [for development] set nil when testing

--K_range	= {9,9} -- to choose K [for development], set nil when testing
--alpha_range = {0.67,0.67} -- to choose alpha (alpha * mstscore + (1-alpha) * iornnscore) [for development] set nil when testing

punc = true -- taking punc into account for evaluation 

