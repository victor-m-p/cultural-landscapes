#!/usr/bin/ruby
# sbatch -N 1 -o BIAS_TESTS/BIAS_TESTS_final -t 24:00:00 -p RM ./bias_correction.rb 20

n=ARGV[0].to_i
label=rand(1 << 20)

list=[]
1000.times {
  beta=rand()**2+0.01
  
  found=false
  count=Hash.new(0)
  running=0
  while(!found and (running < 100)) do  
    `./mpf -g DATA/test_sequence_#{label} #{n} 256 #{beta}`

    file=File.new("DATA/test_sequence_#{label}_data.dat", 'r')
    str=file.read; file.close

    count=Hash.new(0)
    str.split("\n")[2..-1].each { |j| 
      n.times { |k| 
        j.split("")[k] == "1" ? count[k] += 1 : nil 
      } 
    };1
  
    found=(count.to_a.select { |i| i[1] == 128 }.length > 0)
    running += 1
  end
  if (running < 100) then
    type_flip=count.to_a.select { |i| i[1] == 128 }[0][0]

    file=File.new("DATA/test_sequence_#{label}_data.dat", 'r')
    str=file.read; file.close

    str2="#{128+64}\n#{n}\n"
    counter=0
    str.split("\n")[2..-1].each { |obs|
      if obs.split(" ")[0][type_flip] == "0" then
        str2 << obs+"\n"
      else
        if (counter.modulo(2)) == 0 then
          str2 << obs+"\n"
        end
        counter += 1
      end
    }
    file=File.new("DATA/test_sequence_#{label}_bias.dat", 'w')
    file.write(str2); file.close

    str2="#{128+64}\n#{n}\n"
    counter=0
    str.split("\n")[2..-1].each { |obs|
      if obs.split(" ")[0][type_flip] == "0" then
        str2 << obs.split(" ")[0]+" #{0.5}\n"
      else
        if (counter.modulo(2)) == 0 then
          str2 << obs+"\n"
        end
        counter += 1
      end
    }
    file=File.new("DATA/test_sequence_#{label}_reweighted.dat", 'w')
    file.write(str2); file.close


    ## first get the best case
    `./mpf -c DATA/test_sequence_#{label}_data.dat 1`;1
    full=`./mpf -k DATA/test_sequence_#{label}_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

    `./mpf -c DATA/test_sequence_#{label}_bias.dat 1`;1
    bias=`./mpf -k DATA/test_sequence_#{label}_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_bias.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

    `./mpf -c DATA/test_sequence_#{label}_reweighted.dat 1`;1
    reweight=`./mpf -k DATA/test_sequence_#{label}_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_reweighted.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

    ans=["DATA/test_sequence_#{label}_params.dat", "DATA/test_sequence_#{label}_data.dat_params.dat", "DATA/test_sequence_#{label}_bias.dat_params.dat", "DATA/test_sequence_#{label}_reweighted.dat_params.dat"].collect { |trial|
      `./mpf -z #{trial} #{n}`  
      file=File.new(trial+"_probs.dat", 'r')
      count=0
      file.each_line { |line|
        if line.split(" ")[0][type_flip] == "0" then
          count += line.split(" ")[2].to_f
        end
      };
      file.close
      p=count/(1-count)
    }

    kl_list=[full, bias, reweight]
    bias_list=[Math::log(ans[0]/ans[1]), Math::log(ans[0]/ans[2]), Math::log(ans[0]/ans[3])]

    list << [beta, kl_list, bias_list]
    print "#{list}"
    `rm -rf DATA/test_sequence_#{label}*`
  else
    print "Bad Beta!\n"
  end
}
