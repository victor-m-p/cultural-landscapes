#!/usr/bin/ruby  
# NOTE (2026 revision): written for the published version of mpf, whose defaults have since changed
# (marginal objective for missing data, training-rows-only weighted cross-validation searched below
# log-sparsity -2, exact sampling in -g for n <= 24, and NN > 1 now really uses multi-flip neighbours;
# the published code ran NN=1 whatever NN was given). To repeat the published procedure add --legacy
# to the -c / -o / -t / -l calls and --mcmc to the -g calls, and use NN=1. See README.md.
# This script also calls -c / -t with the argument order of an earlier version of the program (a
# log-sparsity before NN), so it will not run as it stands. Kept for the record.

require 'parallel'

class Array
    def ent # naieve entropy from distribution   
      dent=0
      tot=self.sum*1.0
      self.length.times { |i|
        dent = dent - (self[i]/tot)*Math::log2((self[i]+1e-64)/tot)
      }
      dent
    end
end

n_proc=32

n_nodes=20
n_obs=100

set=[]
start=Time.now
1000.times { |pos|

  beta=((rand() < 0.5) ? rand()**2 : rand())

  `./mpf -g test #{n_nodes} #{n_obs} #{beta}`

  file=File.new("test_data.dat", 'r');file.readline;file.readline;count=Hash.new(0);file.each_line { |i| count[i] += 1 };entropy=count.to_a.collect { |i| i[1] }.ent
 
  scan=Array.new(64) { |i| (i-16)/16.0 }
  kl_fit_1=Parallel.map(scan, :in_process=>n_proc) { |logs|
    ans=[logs, `./mpf -t test #{logs} 1`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
    ans
  }

  kl_fit_2=Parallel.map(scan, :in_process=>n_proc) { |logs|
    ans=[logs, `./mpf -t test #{logs} 2`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
    ans
  }
  kl_fit_2[kl_fit_2.index { |i| i[1] == kl_fit_2.collect { |j| j[1] }.min}]

  # kl_fit_3=Parallel.map(scan, :in_process=>n_proc) { |logs|
  #   ans=[logs, `./mpf -t test #{logs} 3`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
  #   ans
  # }
  #
  # kl_fit_3[kl_fit_3.index { |i| i[1] == kl_fit_3.collect { |j| j[1] }.min}]

  loc=scan.index { |i| i == 0 }
  # ans=[beta, entropy, kl_fit_1[0][1], kl_fit_2[0][1], kl_fit_3[0][1], kl_fit_1[loc][1], kl_fit_2[loc][1], kl_fit_3[loc][1], kl_fit_1[kl_fit_1.index { |i| i[1] == kl_fit_1.collect { |j| j[1] }.min}], kl_fit_2[kl_fit_2.index { |i| i[1] == kl_fit_2.collect { |j| j[1] }.min}], kl_fit_3[kl_fit_3.index { |i| i[1] == kl_fit_3.collect { |j| j[1] }.min}]]
  ans=[beta, entropy, kl_fit_1[0][1], kl_fit_2[0][1], kl_fit_1[loc][1], kl_fit_2[loc][1], kl_fit_1[kl_fit_1.index { |i| i[1] == kl_fit_1.collect { |j| j[1] }.min}], kl_fit_2[kl_fit_2.index { |i| i[1] == kl_fit_2.collect { |j| j[1] }.min}]]

  set << ans
  print "#{ans}\n"
  print "#{(Time.now-start)/(pos+1)} per round; finish at #{Time.now + (1000-pos)*(Time.now-start)/(pos+1)}\n"

  file=File.new("saved_sparsity.dat", 'w')
  file.write(Marshal.dump(set))
  file.close
  
}

