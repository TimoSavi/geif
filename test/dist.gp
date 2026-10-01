set terminal pngcairo size 800,800 enhanced font 'Helvetica,10'
set output 'dist.png'
set datafile separator ","
set autoscale fix

plot "dist.csv" using 1:2 with points pt 7 ps 0.5  notitle
