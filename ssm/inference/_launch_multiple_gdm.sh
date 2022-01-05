echo "KILLING PREVIOUS JOBS"
pkill -f _test_fivo.py

tag="gdm-v6-0"

for i in `seq 5`
do
    echo $i

    # python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 0 --proposal-structure 'BOOTSTRAP' --tilt-structure 'NONE'   --log-group "$tag-1-bootstrap" 		> /dev/null 2>&1 &
    python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 0 --proposal-structure 'DIRECT'    --tilt-structure 'NONE'   --log-group "$tag-2-fivo" 		> /dev/null 2>&1 &
    # python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 0 --proposal-structure 'DIRECT'    --tilt-structure 'DIRECT' --log-group "$tag-3-fivo-aux" 		> /dev/null 2>&1 &

    python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'BOOTSTRAP' --tilt-structure 'NONE'   --log-group "$tag-4-bootstrap-sgr"   	> /dev/null 2>&1 &
    # python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'DIRECT'    --tilt-structure 'NONE'   --log-group "$tag-5-fivo-sgr"             > /dev/null 2>&1 &
    python _test_fivo.py --model 'GDM' --seed $i --PLOT 0 --use-sgr 1 --proposal-structure 'DIRECT'    --tilt-structure 'DIRECT' --log-group "$tag-6-fivo-aux-sgr"         > /dev/null 2>&1 &
done

echo "All Done!"
