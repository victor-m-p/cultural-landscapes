#!/usr/bin/ruby
# NOTE (2026 revision): written for the published version of mpf, whose defaults have since changed
# (marginal objective for missing data, training-rows-only weighted cross-validation searched below
# log-sparsity -2, exact sampling in -g for n <= 24, and NN > 1 now really uses multi-flip neighbours;
# the published code ran NN=1 whatever NN was given). To repeat the published procedure add --legacy
# to the -c / -o / -t / -l calls and --mcmc to the -g calls, and use NN=1. See README.md.
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
