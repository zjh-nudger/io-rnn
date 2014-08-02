ROOT_LABEL	= 'ROOT'
NUM_THREADS	= 1

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
DEPSTRUCT_N_DEPS = 100

-- for training
MIN_OCCURS_THRESHOLD	 = 3
TRAIN_LAMBDA			 = 1e-4
TRAIN_LAMBDA_L			 = 1e-10
TRAIN_BATCHSIZE			 = 32
TRAIN_N_LEAPS			 = 100
TRAIN_N_EPOCHS_IN_1_ITER = 5
TRAIN_N_ITER_IN_1_LEAP	 = 5
TRAIN_UPDATE_L			 = true
TRAIN_MST_K_BEST		 = 10
TRAIN_MST_ITERS			 = 1
TRAIN_WEIGHT_MIX		 = 1
TRAIN_N_PROC			 = 10

TRAIN_WEIGHT_LEARNING_RATE	= 0.1
TRAIN_VOCA_LEARNING_RATE	= 0.1

MST_PATH	= "/datastore/phong/data/io-rnn/tools/mstparser/"
CLEAR_PATH	= "/datastore/phong/data/io-rnn/tools/clearnlp/"

TRAIN_MPIR_SENT_LEN = {}
TRAIN_MPIR_N_ITER = {}
for len = 15, 15 do
	TRAIN_MPIR_SENT_LEN[#TRAIN_MPIR_SENT_LEN+1] = len
	if len == 15 then
		TRAIN_MPIR_N_ITER[#TRAIN_MPIR_N_ITER+1] = 100
	else
		TRAIN_MPIR_N_ITER[#TRAIN_MPIR_N_ITER+1] = 1
	end
end
n_phase = #TRAIN_MPIR_SENT_LEN
TRAIN_MPIR_SENT_LEN[n_phase+1] = TRAIN_MPIR_SENT_LEN[n_phase]
TRAIN_MPIR_N_ITER[n_phase+1] = 0


-- for eval
EVAL_TOOL_PATH	= '../tools/eval-dep.pl'
EVAL_EMAIL_ADDR	= 'lephong.xyz@gmail.com'
