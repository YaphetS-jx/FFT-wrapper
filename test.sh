for i in {12..100..1}
do 
    echo $i

    ./test $i $i $i 1 10
    
done


# for i in {1..500..5}
# do 
#     echo $i

#     ./test 50 50 50 1 $i
    
# done