for i in `seq 5`
do
    echo $i

    python _test_fivo.py --model 'LDS' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'RESQ'    --tilt-structure 'NONE'   --log-group 'lds-v0-2-1-fivo' 			> /dev/null 2>&1 &
    python _test_fivo.py --model 'LDS' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'RESQ'    --tilt-structure 'DIRECT' --log-group 'lds-v0-2-0-fivo-aux' 		> /dev/null 2>&1 &
done

echo "All Done!"
