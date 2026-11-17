require 'csv'
load '../../ENT/ent.rb'

file=File.new("../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
properties=Hash.new([])
ans[1..-1].each { |i|
  properties[i[0].to_i] += [i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")]
};1

file=File.new("../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
questions=Hash.new()
questions_order=Hash.new()
count=0
ans[1..-1].each { |i|
  questions[i[0].to_i] = i[1]
  questions_order[count] = i[1]
  count += 1
};1

texts=[1025, 1042, 1050, 1054, 1055, 1056, 1057, 1058, 1062, 1063, 1067, 1074, 1075, 1078, 1079, 1080, 1098, 1100, 1112, 1117, 1120, 1121, 1124, 1141, 1144, 1158, 1160, 1170, 1176, 1177, 1182, 1188, 1203, 1207, 1224, 1232, 1239, 1243, 1252, 1253, 1255, 1260, 1265, 1269, 1270, 1271, 1272, 1290, 1294, 1297, 1324, 1326, 1328, 1330, 1338, 1342, 1345, 1347, 1356, 1367, 1375, 1387, 1389, 1393, 1401, 1405, 1406, 1417, 1418, 1421, 1422, 1424, 1427, 1429, 1432, 1442, 1443, 1452, 1471, 1473, 1484, 1486, 1508, 1509, 1513, 1525]

ans=[texts, properties.keys-texts, properties.keys].collect { |kind|
  counts=Array.new(20) { [] }
  kind.each { |i|
    properties[i].each { |guess|

      20.times { |k|
        if guess[k] != "X" then
          counts[k] << guess[k].to_i*1.0/properties[i].length
        end
      }
      
    }
  }
  counts.collect { |i| [i.mean, i.length > 0 ? (i.var/i.length)**0.5 : 0 ] }
}

20.times { |i|
  print "Q#{i}: #{questions_order[i]}\n"
  print "Text average: #{'%.3f' % ans[0][i][0]} vs. Non-text average: #{'%.3f' % ans[1][i][0]} ± #{'%.3f' % ans[1][i][1]} vs. vs. All: #{'%.3f' % ans[2][i][0]} ± #{'%.3f' % ans[2][i][1]}\n"
}

20.times { |i|
  print "Q#{i}: #{questions_order[i]}\n"
  print "#{'%.3f' % ans[1][i][0]} ± #{'%.3f' % ans[1][i][1]} shifts to #{'%.3f' % ans[2][i][0]} ± #{'%.3f' % ans[2][i][1]}\n"
}
