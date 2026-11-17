require 'CSV'

file=File.new("n20_maxna10.txt_probs.dat", 'r')
lookup=Hash.new()
file.each_line { |i| 
  lookup[i.split(" ")[0]]=i.split(" ")[-1].to_f
};1
file.close

"../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv\n../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv\n../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv\n"
file=File.new("../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
properties=Hash.new([])
ans[1..-1].each { |i|
  properties[i[0].to_i] += [i[1..-2].collect { |j| j.to_i == 0 ? "X" : (j.to_i < 0 ? 0 : 1) }.join("")]
};1

file=File.new("../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
lookup=Hash.new()
ans[1..-1].each { |i|
  lookup[i[0].to_i] = i[1]
};1

file=File.new("../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv", 'r')
csv = CSV.new(file)
ans=csv.read;
file.close
questions=Hash.new()
ans[1..-1].each { |i|
  questions[i[0].to_i] = i[1]
};1

#tag=Hash.new
dates=Hash.new

require "selenium-webdriver"
driver = Selenium::WebDriver.for :safari
wait = Selenium::WebDriver::Wait.new(:timeout => 20)
count=0
text=0
lookup.keys.each { |k|
  if dates[k] == nil then
    driver.navigate.to "https://religiondatabase.org/browse/#{k}/#/"
    sleep(2)
    dates[k]=driver.find_elements(:class, "Date__text--S8V").collect { |i| i.text }[0]
  end
  print "#{k} (#{count}): #{dates[k]}\n"
};1

lookup.keys.each { |k|
  if date[k] == [] then
    driver.navigate.to "https://religiondatabase.org/browse/#{k}/#/"
    sleep(3)
    dates[k]=driver.find_elements(:class, "Date__text--S8V").collect { |i| i.text }[0]
    print "#{k} has #{tag[k]}\n"
  end
};1

file=File.new("DATA/scrape_dates.dat", 'w'); file.write(Marshal.dump(dates)); file.close



