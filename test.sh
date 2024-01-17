for i in {20..200..1}
do 
    echo $i

    ./test $i $i $i 10
    
done


# for i in {1..500..5}
# do 
#     echo $i

#     ./test 50 50 50 1 $i
    
# done