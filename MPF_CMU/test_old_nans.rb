#!/usr/bin/ruby

n=ARGV[0].to_i
nan=ARGV[1].to_i
label=ARGV[2]
`./mpf -g DATA/test_sequence_#{label} #{n} 2048 0.2`

file=File.new("DATA/test_sequence_#{label}_data.dat", 'r')
str=file.read; file.close

file=File.new("DATA/test_sequence_#{label}_base_data.dat", 'w')
str2="128\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

file=File.new("DATA/test_sequence_#{label}_256_data.dat", 'w')
str2="256\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

`./mpf -c DATA/test_sequence_#{label}_base_data.dat 1`
`./mpf -c DATA/test_sequence_#{label}_256_data.dat 1`
start=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_base_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
best=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_256_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

cut=1024
str_na=str.split("\n")[1..(128+1)].join("\n")+"\n"+str.split("\n")[130..(129+cut+1)].collect { |j| 
  loc=[]
  while(loc.length < nan) do
    while(loc.include?(pos=rand(n))) do     
    end
    loc << pos    
  end
  code=j.dup
  loc.each { |i|
    code[i]="X"
  }
  code
}.join("\n");1

[64, 128, 256, 512, 512+256, 1024].each { |cut| #, 512, 512+256, 1024
  file=File.new("DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat", 'w')
  file.write("#{128+cut}\n"+str_na); file.close
  `./mpf -c DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat 1`  
  ans=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
  print "#{cut}: #{ans} (vs #{best}, vs #{start})\n"
}

print "Now do bad choice...\n"
avg=Array.new(n) { 0 }
str.split("\n")[2..(128+1)].each { |i|
  set=i.split(" ")[0].split("")
  n.times { |k|
    avg[k] += set[k].to_f
  }
};
avg.collect! { |i| i/128.0 }
cut=1024
str_na_new=str_na.split("\n")[1..(128+1)].join("\n")+"\n"+str_na.split("\n")[130..(129+cut+1)].collect { |j| 
  code=j.dup
  Array.new(n) { |i| code[i] == "X"  ? avg[i].round.to_s : code[i] }.join("")+" 1.0"
}.join("\n");1

[64, 128, 256, 512, 512+256, 1024].each { |cut| #, 512, 512+256, 1024
  file=File.new("DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat", 'w')
  file.write("#{128+cut}\n"+str_na_new); file.close
  `./mpf -c DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat 1`  
  ans=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
  print "#{cut}: #{ans} (vs #{best}, vs #{start})\n"
}
