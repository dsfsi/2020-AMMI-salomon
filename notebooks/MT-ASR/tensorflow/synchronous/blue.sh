hyp=$1
ref=tst2015.fr
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
