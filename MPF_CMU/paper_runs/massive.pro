;Iteration 0 256 0.5 (90.412224429 sec.): 0.325617 vs 3.425569
it=0
ndat=256
kl_nosparsity=3.425569
kl_neg0=0.325617 

;Iteration 5 128 0.25 (36.45584924 sec.): 0.58513 vs 2.024822
;Iteration 7 128 0.25 (64.419422083 sec.): 0.654483 vs 6.524502
; it=7
; ndat=128
; kl_nosparsity=6.524502
; kl_neg0=0.654483

;Iteration 0 512 0.5 (176.744288931 sec.): 0.138966 vs 0.876074
; it=0
; ndat=512
; kl_nosparsity=0.876074
; kl_neg0=0.138966


;; close match to the observed
name='test_'+strn(it)+'_'+strn(ndat)+'DATA'

openr, 1, 'TEST_COMP/'+name+'_params.dat_probs.dat'
;openr, 1, 'TEST/test_params.dat_probs.dat'
n=2l^20
true=dblarr(3, n)
readf, 1, true
true=reform(true[2,*],n)
close, 1

openr, 1, 'TEST_COMP/'+name+'_data.dat_params.dat_probs_CV.dat'
;openr, 1, 'TEST/test_data.dat_params.dat_probs_CV.dat'
n=2l^20
cv=dblarr(3, n)
readf, 1, cv
cv=reform(cv[2,*],n)
close, 1

openr, 1, 'TEST_COMP/'+name+'_data.dat_params.dat_probs.dat'
;openr, 1, 'TEST/test_data.dat_params.dat_probs.dat'
n=2l^20
no=dblarr(3, n)
readf, 1, no
no=reform(no[2,*],n)
close, 1

w=where(no lt 2e-20)
no[w]=2e-20

;; subsample!
; w=(shuffle(lindgen(n)))[0:n/10-1]
; no=no[w]
; true=true[w]
; cv=cv[w]

c = FINDGEN(17) * (!PI*2/16.)  
s=0.2
USERSYM, s*COS(c), s*SIN(c), /FILL

set_plot, 'ps'
device, filename='PLOTS/sample_fit_new_'+strn(ndat)+'.eps', SET_FONT='Helvetica', /TT_FONT, bits_per_pixel=8
colors=[cgColor('Dodger Blue', 1), cgColor('Sea Green', 2), cgColor('Crimson', 3), cgColor('Dark Goldenrod', 4), cgColor('Charcoal', 5), cgColor('Light Cyan', 6)]

;xrange=[1e-8, 1], yrange=[1e-20, 1]
plot, true, cv, /xlog, /ylog, xrange=[1e-8, 1], yrange=[1e-20, 1], psym=4, xtitle='True Probabilities', ytitle='Inferred Probabilities', /nodata, font=1, xthick=2, ythick=2, charthick=2, charsize=1.5
oplot, [1d-20,1], [1d-20,1], thick=2

oplot, true, cv, psym=8, color=1

oplot, true, no, psym=8, color=3
oplot, true, no, psym=7, color=3, SymSize=0.25

xyouts, 1.5e-3, 1e-8, 'KL, Standard MPF: '+strn(kl_nosparsity)+' ', color=3, font=1
xyouts, 1.5e-3, 1e-7, 'KL, Regularized MPF: '+strn(kl_neg0)+' ', color=1, font=1


device, /close_file
set_plot, 'x'

stop
end