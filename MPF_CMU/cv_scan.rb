#!/opt/local/bin/ruby
#!/usr/bin/ruby
# sbatch -N 100 -o CV_SCAN --mail-type=ALL -t 12:00:00 -p RM ./cv_scan.rb

require 'parallel'
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

n_proc=8

n_nodes=20
n_obs=100

set=[]
start=Time.now

file=File.new("saved_cv_tests.dat", 'r')
set=Marshal.load(file.read)
file.close

#Parallel.map(Array.new(800) { |i| i }, :in_process=>200) 
(1000-set.length).times { |pos|

  beta=((rand() < 0.5) ? rand()**2 : rand())
  if (beta < 0.1) then
    scan=Array.new(64) { |i| (i-1)/16.0 }    
  else
    scan=Array.new(64) { |i| (i-15)/16.0 }        
  end
  `./mpf -g test #{n_nodes} #{n_obs} #{beta}`
  scan[0]=-100

  file=File.new("test_data.dat", 'r');file.readline;file.readline;count=Hash.new(0);file.each_line { |i| count[i] += 1 };entropy=count.to_a.collect { |i| i[1] }.ent

  cv_test=Parallel.map(scan, :in_process=>n_proc) { |logs|
    val=`./mpf -c test_data.dat #{logs} 1`.scan(/point:[^\n]+\n/).collect { |i| i.split(" ")[-1].to_f }.mean
    ans_cv=[logs, val]
    ans_true=[logs, `./mpf -t test #{logs} 1`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
    [ans_cv, ans_true]
  }

  cv_set=cv_test.transpose[0]
  true_set=cv_test.transpose[1]

  loc=cv_set.index { |i| i[1] == cv_set.collect { |j| j[1] }.max }
  loc_best=true_set.index { |i| i[1] == true_set.collect { |j| j[1] }.min }
  loc_one=cv_set.index { |i| i[0] == 1.0 }

  begin
    ans=[beta, entropy, true_set[0], true_set[loc_one], [cv_set[loc][0], true_set[loc][1]], true_set[loc_best]]
    print "#{ans}\n"
  rescue
    print "#{loc_one} #{loc_best} #{loc}\n"
    print "#{cv_set}\n"
    print "#{true_set}\n"
  end
  
  set << ans
  file=File.new("saved_cv_tests.dat", 'w')
  file.write(Marshal.dump(set))
  file.close
 
  print "#{(Time.now-start)/(pos+1)} per round; finish at #{Time.now + (1000-pos)*(Time.now-start)/(pos+1)}\n"
  # `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "new cross-validated fits (from PSC)" ; git push`
}
