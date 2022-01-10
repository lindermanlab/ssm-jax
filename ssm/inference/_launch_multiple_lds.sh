echo "KILLING PREVIOUS JOBS"
pkill -f _test_fivo.py

tag="lds-v1-0"

for i in `seq 5`
do
    echo $i

    python _test_fivo.py --model 'LDS' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'RESQ'    --tilt-structure 'NONE'   --log-group "$tag-0-fivo" 				> /dev/null 2>&1 &
    python _test_fivo.py --model 'LDS' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'RESQ'    --tilt-structure 'DIRECT' --log-group "$tag-1-fivo-aux" 				> /dev/null 2>&1 &
    python _test_fivo.py --model 'LDS' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'RESQ'    --tilt-structure 'DIRECT' --log-group "$tag-2-fivo-aux-temper" --temper 5.0 	> /dev/null 2>&1 &
done

echo "All Done!"
