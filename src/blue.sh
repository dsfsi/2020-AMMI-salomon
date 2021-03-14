hyp=$1
ref=$hyp.ref
# ../../../../data/external/LiSTra/train_models_enln/rapport/wait-2-0.28/test.ln.output.ref
sed -r 's/(@@ )|(@@ ?$)//g' $hyp > $hyp.out
sed -r 's/(@@ )|(@@ ?$)//g' $ref > $ref.out
# remove the delay symbol
sed -i 's/.\{30\}//' $hyp.out
sed -i 's/.\{30\}//' $ref.out
#perl chi_char_segment.pl -t plain < $hyp.out > $hyp.seg
#perl chi_char_segment.pl -t plain < $ref.out > $ref.seg
#mv $hyp.seg $hyp.out
#mv $ref.seg $ref.out
perl multi-bleu.perl $ref.out < $hyp.out
rm $hyp.out
rm $ref.out

# bash blue.sh ../../../../data/external/LiSTra/train_models_enln/rapport/wait-2-0.28/test.en.output

# bash blue.sh ../../../../data/external/LiSTra/train_models_enln/rap
# port/wait-2-0.28/test.ln.output
