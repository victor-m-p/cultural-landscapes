# NOTE (2026 revision): written for the published version of mpf, whose defaults have since changed
# (marginal objective for missing data, training-rows-only weighted cross-validation searched below
# log-sparsity -2, exact sampling in -g for n <= 24, and NN > 1 now really uses multi-flip neighbours;
# the published code ran NN=1 whatever NN was given). To repeat the published procedure add --legacy
# to the -c / -o / -t / -l calls and --mcmc to the -g calls, and use NN=1. See README.md.
require 'parallel'

preface="/Users/simon/Desktop/humanities-glass/data/clean/"
preface_new="/Users/simon/Desktop/humanities-glass/data/mdl/"

set=`ls -alh /Users/simon/Desktop/humanities-glass/data/clean`.split("\n").collect { |i| i.split(" ")[-1] }.select { |i| i.include?("txt") and !i.include?("mpf") }.collect { |i| 
  n=i.split("_")[6].to_i
  na=i.split("_")[-1].to_i
  [n, na, i]
}.sort { |i,j| (i[0] <=> j[0]) == 0 ? i[1] <=> j[1] : i[0] <=> j[0] }

[1,2,3].each { |nn|

  Parallel.map(set, :in_process=>8) { |trial|
    n_lines=`wc -l #{preface+trial[-1]}`.split(" ")[0].to_i
    n=trial[0]
    file=File.new(preface+trial[-1], 'r')
    str=""
    file.each_line { |set|
      str << set.split(" ")[0..-2].collect { |i| (i.to_i == 0) ? "X" : ((i.to_i < 0) ? "0" : "1") }.join()+" "+set.split(" ")[-1]+"\n"
    }
    file.close
  
    file_out=File.new(preface_new+trial[-1]+".mpf", 'w')
    file_out.write("#{n_lines}\n#{n}\n")
    file_out.write(str); file_out.close
  
      ans=`./mpf -l #{preface_new+trial[-1]+".mpf"} 0 #{nn}`
      ans.split("\n").select { |i| i.include?("params") }
      file_out=File.new(preface_new+trial[-1]+".mpf_params_NN#{nn}", 'w')
      file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
      file_out.close
  
      print "n=#{trial[0]}; maxNA=#{trial[1]}; NN=#{nn}: #{ans.split("\n").select { |i| i.include?("Clock") }[0]}\n"    
  }
  `cd /Users/simon/Desktop/humanities-glass ; git add . ; git commit -m "updating files #{Time.now}" ; git push`
}
