dic=$1
model=$2

echo "===================== SCWS ================="
th eval_scws.lua $dic ../data/SCWS/ $model context

echo "==================== ML compound ==================="
th eval_ml_compound.lua $dic ../data/ML/compounds.txt $model

echo "====================== lex sub ================="
th eval_lexsubtask.lua $dic ../data/lex_sub_task/test $model context 
