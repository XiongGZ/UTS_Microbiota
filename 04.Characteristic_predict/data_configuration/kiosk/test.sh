awk -F / '{print $(NF-1)"\t"$NF}' Sfilepath | sort > Source
sed -i "1i city\tsampleid" Source

for i in {0..9};do
  awk -F / '{print $(NF-1)"\t"$NF}' Qfilepath$i | sort > Query$i
  sed -i "1i city\tsampleid" Query$i
  awk -F / '{print $(NF-1)"\t"$NF}' Tfilepath$i | sort > Transfer$i
  sed -i "1i city\tsampleid" Transfer$i
done

rm *filepath*