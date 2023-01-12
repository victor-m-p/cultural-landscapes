#!/usr/bin/ruby
# sbatch -N 1 -o CLEANED_FITS_3 -t 12:00:00 -p RM ./cleaned_fits.rb

`ls ../data/mdl_final/all_*`.split("\n").each { |file|
  if (file.scan(/maxna_[0-9]+/)[0].split("_")[-1].to_i > 5) then
    start=Time.now
    print "Doing #{file}...\n"
    print `OMP_NUM_THREADS=128 ./mpf -c #{file} 1`  
    print "#{Time.now-start} seconds\n"

    `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "cleaned with-text fits (from PSC)" ; git push`
  end
}
