#!/usr/bin/ruby
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 1 1
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 2 1
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 3 1
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 4 1
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 5 1
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 1 2
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 2 2
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 3 2
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 4 2
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb 5 2

list=[]
start=Time.now
10000.times { |pos|
  beta=0
  while(beta < 0.01) do
    beta=rand()**2
  end
  
  `./mpf -g test_#{ARGV[0]}_#{ARGV[1]} 20 128 #{beta}`
  ans=`./mpf -o test_#{ARGV[0]}_#{ARGV[1]} #{ARGV[1]}`
  begin
    val=[beta]+eval(ans.split("\n").select { |i| i.include?("val=") }[0].split("=")[-1].gsub("inf", "Float::INFINITY"))
  rescue
    val=[]
  end
  list << val
  print "#{(Time.now-start)/(pos+1)} per loop.\n"
  print "#{val}\n"
  file=File.new("cv_full_#{ARGV[0]}_#{ARGV[1]}NN.dat", 'w')
  file.write(Marshal.dump(list))
  file.close  
}
