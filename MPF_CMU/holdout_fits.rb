#!/usr/bin/ruby
# sbatch -N 1 -o HOLDOUT_FITS_2 -t 03:00:00 -p RM ./holdout_fits.rb

`ls ../data/mdl_final/holdout_*`.split("\n").each { |file|
  if (file.scan(/maxna_[0-9]+/)[0].split("_")[-1].to_i >= 4) then
    start=Time.now
    print "Doing #{file}...\n"
    print `OMP_NUM_THREADS=128 ./mpf -c #{file} 1`  
    print "#{Time.now-start} seconds\n"

    `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "holdout fits (from PSC)" ; git push`
  end
}
