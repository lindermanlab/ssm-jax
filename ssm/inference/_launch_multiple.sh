for i in `seq 10`
do
    echo $i
    python _test_fivo.py --seed $i > /dev/null 2>&1 &
done
echo "All Done!"
