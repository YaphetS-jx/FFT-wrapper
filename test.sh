# for i in {10..300..5}
# do 
#     echo $i

#     ./test_mc 100 100 100 10 $i
    
# done

# for i in {20..300..2}
# do 
#     echo $i

#     ./test_mc $i $i $i 10 10 
    
# done



grep 'FFT_iFFT_CPU' result* | cut -f3 -d: | cut -f1 -dm | tee FFT_iFFT_CPU
grep 'FFT_iFFT_GPU' result* | cut -f3 -d: | cut -f1 -dm | tee FFT_iFFT_GPU
grep 'FFT_iFFT_GPU' result* | cut -f3 -d: | cut -f3 -ds | cut -f1 -dm | tee FFT_iFFT_GPU_c

grep 'FFT_iFFT_complex_CPU' result* | cut -f3 -d: | cut -f1 -dm | tee FFT_iFFT_complex_CPU
grep 'FFT_iFFT_complex_GPU' result* | cut -f3 -d: | cut -f1 -dm | tee FFT_iFFT_complex_GPU
grep 'FFT_iFFT_complex_GPU' result* | cut -f3 -d: | cut -f3 -ds | cut -f1 -dm | tee FFT_iFFT_complex_GPU_c

grep 'Kron_single_col_CPU' result* | cut -f3 -d: | cut -f1 -dm | tee Kron_single_col_CPU
grep 'Kron_single_col_GPU' result* | cut -f3 -d: | cut -f1 -dm | tee Kron_single_col_GPU
grep 'Kron_single_col_GPU' result* | cut -f3 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_single_col_GPU_c

grep 'Kron_single_col_complex_CPU' result* | cut -f3 -d: | cut -f1 -dm | tee Kron_single_col_complex_CPU
grep 'Kron_single_col_complex_GPU' result* | cut -f3 -d: | cut -f1 -dm | tee Kron_single_col_complex_GPU
grep 'Kron_single_col_complex_GPU' result* | cut -f3 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_single_col_complex_GPU_c


grep 'Kron_multiple_col_CPU' result_mc1 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_CPU_1
grep 'Kron_multiple_col_GPU' result_mc1 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_GPU_1
grep 'Kron_multiple_col_GPU' result_mc1 | cut -f2 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_multiple_col_GPU_1_c


grep 'kron_multiple_col_complex_CPU' result_mc1 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_complex_CPU_1
grep 'kron_multiple_col_complex_GPU' result_mc1 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_complex_GPU_1
grep 'kron_multiple_col_complex_GPU' result_mc1 | cut -f2 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_multiple_col_complex_GPU_1_c


grep 'Kron_multiple_col_CPU' result_mc2 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_CPU_2
grep 'Kron_multiple_col_GPU' result_mc2 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_GPU_2
grep 'Kron_multiple_col_GPU' result_mc2 | cut -f2 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_multiple_col_GPU_2_c

grep 'kron_multiple_col_complex_CPU' result_mc2 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_complex_CPU_2
grep 'kron_multiple_col_complex_GPU' result_mc2 | cut -f2 -d: | cut -f1 -dm | tee Kron_multiple_col_complex_GPU_2
grep 'kron_multiple_col_complex_GPU' result_mc2 | cut -f2 -d: | cut -f3 -ds | cut -f1 -dm | tee Kron_multiple_col_complex_GPU_2_c