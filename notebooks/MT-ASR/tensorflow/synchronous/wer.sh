hyp=$1
ref=$hyp.ref #../data/raw_data/tst2015.en
sed -r 's/(@@ )|(@@ ?$)//g' $hyp > $hyp.out
sed -r 's/(@@ )|(@@ ?$)//g' $ref > $ref.out
sed -i 's/.\{6\}//' $hyp.out
sed -i 's/.\{6\}//' $ref.out
python wer.py $ref.out  $hyp.out
rm $hyp.out
rm $ref.out
