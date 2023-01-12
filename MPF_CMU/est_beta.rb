load '../../ENT/ent.rb'
set=`ls ../data/mdl/*nuniq_20*LAM*`.split("\n").collect { |i|
  file=File.new(i, 'r')
  params=eval(file.read)
  file.close
  [params.mean, params.var]
}.transpose[1]

set.mean**0.5
((set.var/set.length)**0.5)**0.5
