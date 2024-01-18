for i in {30..300..1}
do 
    echo $i

    ./test $i $i $i 10 10 
    
done