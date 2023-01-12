#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o UPDATED_FITS -t 48:00:00 -p RM ./final_process.rb

prefix="/jet/home/sdedeo/humanities-glass/data/clean/"
new_prefix="/jet/home/sdedeo/humanities-glass/data/mdl_experiments/"

`ls #{prefix}`.split("\n").each { |filename|
  print "Doing #{filename}\n"
  filename_out=filename+".mpf"
  
  n_lines=`wc -l #{prefix+filename}`.split(" ")[0].to_i
  n=filename.split("_")[2].to_i
  num_na=filename.split("_")[4].to_i
  
  if (n == 20) and (num_na == 5) then
  
    file=File.new(prefix+filename, 'r')
    str=""
    file.each_line { |set|
      str << set.split(" ")[0..-2].collect { |i| (i.to_i == 0) ? "X" : ((i.to_i < 0) ? "0" : "1") }.join()+" "+set.split(" ")[-1]+"\n"
    }
    file.close
    file_out=File.new(new_prefix+filename_out, 'w')
    file_out.write("#{n_lines}\n#{n}\n")
    file_out.write(str); file_out.close

    ans=`OMP_NUM_THREADS=128 ./mpf -c #{new_prefix+filename_out} 1`
    print "FINAL VERSION for #{filename_out}:\n #{ans}\n\n"
    ans.split("\n").select { |i| i.include?("params") }
    file_out=File.new(new_prefix+filename_out+"_params_NN1", 'w')
    file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
    file_out.close
  
    `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "new cross-validated fits (from PSC)" ; git push`
    
  end
}
