#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o UPDATED_FITS -t 24:00:00 -p RM ./best_fit_DRH_2.rb

class Array
  def mean
    self.sum*1.0/self.length
  end
end

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

require 'parallel'
n_proc=32+16

preface="../data/clean/"
preface_new="../data/mdl/"

set=`ls -alh ../data/clean`.split("\n").collect { |i| i.split(" ")[-1] }.select { |i| i.include?("txt") and !i.include?("mpf") }.collect { |i| 
  n=i.split("_")[6].to_i
  na=i.split("_")[-1].to_i
  [n, na, i]
}.sort { |i,j| (i[0] <=> j[0]) == 0 ? i[1] <=> j[1] : i[0] <=> j[0] }

set.select { |i| i[-1].include?("nuniq_20") }.each { |trial|

  nn=1
  scan=Array.new(32+16) { |i| (i-8)/16.0 }     

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

  file=File.new(preface_new+trial[-1]+".mpf", 'r');file.readline;file.readline;count=Hash.new(0);file.each_line { |i| count[i] += 1 };entropy=count.to_a.collect { |i| i[1] }.ent

  cv_test=Parallel.map(scan, :in_process=>n_proc) { |logs|
    val=`OMP_NUM_THREADS=128 ./mpf -c #{preface_new+trial[-1]+".mpf"} #{logs} #{nn}`.scan(/point:[^\n]+\n/).collect { |i| i.split(" ")[-1].to_f }.mean
    ans_cv=[logs, val]
    print "#{ans_cv}\n"
    ans_cv
  }
  loc=cv_test.index { |i| i[1] == cv_test.collect { |j| j[1] }.max }
  best_lambda=cv_test[loc][0]

  print "Best lambda for #{trial[-1]}: #{best_lambda}\n"

  ans=`./mpf -l #{preface_new+trial[-1]+".mpf"} #{best_lambda} #{nn}`
  ans.split("\n").select { |i| i.include?("params") }
  file_out=File.new(preface_new+trial[-1]+".mpf_params_NN#{nn}_#{best_lambda}lambda_UPDATED", 'w')
  file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
  file_out.close
  
  `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "new cross-validated fits (from PSC)" ; git push`
}

