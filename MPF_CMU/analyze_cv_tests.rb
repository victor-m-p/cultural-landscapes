#`git pull`;1
load '../../ENT/ent.rb'

class Float
  
  def r
    (self*100).round/100.0
  end
end

[1,2].each { |nn|
  list=[]
  `ls DATA/cv_full_*_#{nn}NN.dat`.split("\n").each { |fn|
    file=File.new(fn, 'r')
    list += Marshal.load(file.read).select { |i| i.length > 0 }
    file.close
  }

  print "$\\beta$ range & Optimal KL & KL with CV & KL without sparsity \\\\ \\hline\n"
  [[0.01,0.125, "dispersed"], [0.125, 0.25, "ordered"], [0.25, 0.5, "near-critical"], [0.5, 1.0, "critical"]].each { |range|
    set=list.select { |i| i[0] > range[0] and i[0] < range[1] }.select { |i| i[7].finite? }
    ent=set.transpose[1].mean.r
    best=set.transpose[3].mean.r
    best_loc=set.transpose[4].mean.r
    cv=set.transpose[5].mean.r
    cv_loc=set.transpose[6].mean.r
    no_sparse=set.transpose[7].mean.r
  
    # print "#{range} (#{set.length}): #{best} (#{ent}) (#{best_loc}) #{cv} (#{cv_loc}) #{no_sparse}\n"
    print "#{range[0]}--#{range[1]} (#{range[2]}) & #{'%.2f' % best} & #{'%.2f' % cv} & #{'%.1f' % no_sparse} \\\\ \n"
  }
  print "\n"
}
