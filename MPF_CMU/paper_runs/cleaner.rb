require 'CSV'

list=[1025, 1042, 1050, 1054, 1055, 1056, 1057, 1058, 1062, 1063, 1067, 1074, 1075, 1078, 1079, 1080, 1098, 1100, 1112, 1117, 1120, 1121, 1124, 1141, 1144, 1158, 1160, 1170, 1176, 1177, 1182, 1188, 1203, 1207, 1224, 1232, 1239, 1243, 1252, 1253, 1255, 1260, 1265, 1269, 1270, 1271, 1272, 1290, 1294, 1297, 1324, 1326, 1328, 1330, 1338, 1342, 1345, 1347, 1356, 1367, 1375, 1387, 1389, 1393, 1401, 1405, 1406, 1417, 1418, 1421, 1422, 1424, 1427, 1429, 1432, 1442, 1443, 1452, 1471, 1473, 1484, 1486, 1508, 1509, 1513, 1525]

file=File.new("../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close

0.upto(10) { |na|
  set=[]
  ans[1..-1].each { |i|
    if !list.include?(i[0].to_i) then
      str=i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")
      if str.scan(/X/).length <= na then
        set << [i[0], str, i[-1].to_f]
      end
    end
  };1

  file=File.new("../data/mdl_final/cleaned_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.length}\n#{20}\n#{set.collect { |k| k[1..-1].join(" ") }.join("\n")}")
  file.close

  file=File.new("../data/mdl_final/reference_with_entry_id_cleaned_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.collect { |k| k.join(" ") }.join("\n")}")
  file.close
  
  print "#{na}: #{set.length}\n"
}

0.upto(10) { |na|
  set=[]
  ans[1..-1].each { |i|
    str=i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")
    if str.scan(/X/).length <= na then
      set << [i[0], str, i[-1].to_f]
    end
  };1

  file=File.new("../data/mdl_final/all_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.length}\n#{20}\n#{set.collect { |k| k[1..-1].join(" ") }.join("\n")}")
  file.close

  file=File.new("../data/mdl_final/reference_with_entry_id_all_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.collect { |k| k.join(" ") }.join("\n")}")
  file.close
  
  print "#{na}: #{set.length}\n"
}

file=File.new("../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
properties=Hash.new([])
ans[1..-1].each { |i|
  properties[i[0].to_i] += [i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")]
};1

keep=properties.keys.shuffle[0..(properties.keys.length-list.length-1)]

0.upto(10) { |na|
  set=[]
  ans[1..-1].each { |i|
    if keep.include?(i[0].to_i) then
      str=i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")
      if str.scan(/X/).length <= na then
        set << [i[0], str, i[-1].to_f]
      end
    end
  };1

  file=File.new("../data/mdl_final/holdout_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.length}\n#{20}\n#{set.collect { |k| k[1..-1].join(" ") }.join("\n")}")
  file.close

  file=File.new("../data/mdl_final/reference_with_entry_id_holdout_nrows_#{set.length}_maxna_#{na}.dat", 'w')
  file.write("#{set.collect { |k| k.join(" ") }.join("\n")}")
  file.close
  
  print "#{na}: #{set.length}\n"
}
