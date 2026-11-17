class Integer
  def rep
    rep=self.to_s(2)
    rep="0"*(20-rep.length)+rep
    rep
  end
end

file=File.new("../data/analysis/configuration_probabilities.txt", 'r')
set=Hash.new
(1 << 20).times { |i|
  rep=i.to_s(2)
  rep="0"*(20-rep.length)+rep
  set[rep]=file.readline.to_f
}
file.close


pos=rand(1 << 20).rep

best_pos="11111010011110000110"

set.to_a.sort { |i,j| j[1] <=> i[1] }[0..19].collect { |i|
  best_pos=i[0]
  alts=[set[best_pos]]+Array.new(20) { |i|
    temp=best_pos.dup
    temp[i]=((temp[i] == "0") ? "1" : "0")
    set[temp]
  }
  alts=alts.collect { |i| i/alts.sum }
  alts=alts[0..0]+alts[1..-1].sort.reverse

  Array.new(20) { |i|
    alts[0..i].sum
  }
}

