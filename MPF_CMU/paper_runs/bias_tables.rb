load '../../ENT/ent.rb'

list=[]
`ls BIAS_TESTS`.split("\n").collect { |i| 
	file=File.new("BIAS_TESTS/"+i, 'r'); str=file.read; file.close; 
  str=str.split("slurmstepd")[0]
  if !str.include?("eta") then 
    list += eval(str.split("]]]")[-1]+"]]]")
  end 
}

range=[[0.01,0.125, "dispersed"], [0.125, 0.25, "ordered"], [0.25, 0.5, "near-critical"], [0.5, 1.0, "critical"]]
range.each { |i|
  set=list.select { |j| j[0] > i[0] and j[0] < i[1] }
  
  ans=set.collect { |k| k[1] }.transpose
  ans2=set.collect { |k| k[2] }.transpose
  
  print "#{i[0]}--#{i[1]} (#{i[2]}) &  #{'%.2f' % ans[0].mean} & #{'%.2f' % ans[2].mean} & #{'%.2f' % ans[1].mean} & #{'%.1f' % ((Math::exp(ans2[2].mean)-1)*100)}\% & #{'%.1f' % ((Math::exp(ans2[1].mean)-1)*100)}\% \\\\ \n"
};1
