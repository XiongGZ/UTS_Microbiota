#!/usr/bin/bash

for i in {81..86};do

  cd $i

  for j in {0..9};do
    python /data4/xiongguangzhou/02.Transit/08.Transfer_RF/transferRF.py SourceCM.csv TransferCM$j.csv QueryCM$j.csv $j > result$j
  done

  cd ..

done
