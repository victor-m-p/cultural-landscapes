load '../../ENT/ent.rb'

list=[]
(Dir.glob("NAN_TESTS/*exp*_20_5_*").collect { |f|
  file=File.open(f, 'r'); str=file.read; file.close
  if str.include?("[[") then
    ans=eval(str.scan(/^\[\[[^\n]+\n/)[-1])
  else
    nil
  end
}-[nil]).each { |i|
  list += i
}

`git pull`
range=[[0.01,0.125, "dispersed"], [0.125, 0.25, "ordered"], [0.25, 0.5, "near-critical"], [0.5, 1.0, "critical"]]
range.each { |i|
  set=list.select { |j| j[0] > i[0] and j[0] < i[1] }.transpose
  print "#{i[0]}--#{i[1]} (#{i[2]}) (#{set.transpose.length}) &  #{'%.2f' % set[1].mean} ± #{'%.2f' % (set[1].var/set[1].length)**0.5} & #{'%.2f' % set[3].mean} & #{'%.2f' % set[4].mean} & #{'%.2f' % set[2].mean} \\\\ \n"
};1
