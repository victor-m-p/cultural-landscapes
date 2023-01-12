#!/usr/bin/ruby

print `ls`.split("\n").collect { |i| 
	file=File.new(i, 'r'); str=file.read; file.close; if !str.include?("slurm") and !str.include?("eta") then eval(str.split("]]]")[-1]+"]]]").length; end 
}.select { |i| i != nil }.sum
