64: 0.525828 (vs 0.351919 vs 0.209386 vs 0.617184)
128: 0.473472 (vs 0.351919 vs 0.209386 vs 0.617184)
256: 0.407867 (vs 0.351919 vs 0.209386 vs 0.617184)
512: 0.274963 (vs 0.351919 vs 0.209386 vs 0.617184)
768: 0.23741 (vs 0.351919 vs 0.209386 vs 0.617184)
1024: 0.200955 (vs 0.351919 vs 0.209386 vs 0.617184)
Now do bad choice...
64: 0.59462 (vs 0.351919 vs 0.209386 vs 0.617184)
128: 0.56937 (vs 0.351919 vs 0.209386 vs 0.617184)
256: 0.615963 (vs 0.351919 vs 0.209386 vs 0.617184)
512: 0.624162 (vs 0.351919 vs 0.209386 vs 0.617184)
768: 0.659886 (vs 0.351919 vs 0.209386 vs 0.617184)
1024: 0.665563 (vs 0.351919 vs 0.209386 vs 0.617184)

file=File.new("DATA/NAN_TESTS_20nodes_5NAN_1_DDLONG", 'r')
str=file.read; file.close
ans=str.split("sbatch")[0].split("Now do bad choice...\n").collect { |i|
  i.split("\n").collect { |j| [j.split(":")[0].to_f, j.split(":")[1].to_f, j.split("vs")[1].to_f, j.split("vs")[2].to_f, j.split("vs")[3].to_f] }
};
str="good=#{[[0,ans[0][0][4], ans[0][0][2], ans[0][0][3], ans[0][0][4]]]+ans[0]}\nbad=#{[[0,ans[1][0][4], ans[1][0][2], ans[1][0][3], ans[1][0][4]]]+ans[1]}\n"
print str
