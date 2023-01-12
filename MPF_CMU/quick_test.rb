#!/usr/bin/ruby
# sbatch -N 1 -o quick_test_COMP_FINAL -t 12:00:00 -p RM ./quick_test.rb 

list=[]
[128, 256, 512].each { |n_data|
  ans=Array.new(10) { |i|
    if (i.modulo(2) == 0) then
      beta=0.5
    else
      beta=0.25
    end
    start=Time.now
    
    `./mpf -g TEST_COMP/test_#{i}_#{n_data}DATA 20 #{n_data} #{beta}`

    `./mpf -z TEST_COMP/test_#{i}_#{n_data}DATA_params.dat 20`

    `./mpf -c TEST_COMP/test_#{i}_#{n_data}DATA_data.dat 1`
  
    `./mpf -z TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat 20`

    `cp TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat_probs.dat TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat_probs_CV.dat`
    `cp TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params_CV.dat`

    `./mpf -l TEST_COMP/test_#{i}_#{n_data}DATA_data.dat -100 1`

    `./mpf -z TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat 20`

    cv_performance=`./mpf -k TEST_COMP/test_#{i}_#{n_data}DATA_data.dat TEST_COMP/test_#{i}_#{n_data}DATA_params.dat TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params_CV.dat`.split("\n")[0].split(":")[-1].to_f
    baseline_performance=`./mpf -k TEST_COMP/test_#{i}_#{n_data}DATA_data.dat TEST_COMP/test_#{i}_#{n_data}DATA_params.dat TEST_COMP/test_#{i}_#{n_data}DATA_data.dat_params.dat`.split("\n")[0].split(":")[-1].to_f

    print "Iteration #{i} #{n_data} #{beta} (#{Time.now-start} sec.): #{cv_performance} vs #{baseline_performance}\n"
  }  
}

# ./mpf -k TEST_COMP/test_2_256DATA_data.dat TEST_COMP/test_2_256DATA_params.dat TEST_COMP/test_2_256DATA_data.dat_params.dat
# ./mpf -z TEST_COMP/test_2_256DATA_data.dat_params.dat
