for i in `seq 5`
do
    echo $i

    % python _test_fivo.py --seed $i --use-sgr 0 --proposal-structure 'BOOTSTRAP' --tilt-structure 'NONE'   --log-group 'gdm-v3.0.1-bootstrap' 		> /dev/null 2>&1 &
    python _test_fivo.py --seed $i --use-sgr 0 --proposal-structure 'DIRECT'    --tilt-structure 'NONE'   --log-group 'gdm-v3.0.2-fivo' 		> /dev/null 2>&1 &
    python _test_fivo.py --seed $i --use-sgr 0 --proposal-structure 'DIRECT'    --tilt-structure 'DIRECT' --log-group 'gdm-v3.0.3-fivo-aux' 		> /dev/null 2>&1 &

    python _test_fivo.py --seed $i --use-sgr 1 --proposal-structure 'BOOTSTRAP' --tilt-structure 'NONE'   --log-group 'gdm-v3.0.4-bootstrap-sgr'   	> /dev/null 2>&1 &
    % python _test_fivo.py --seed $i --use-sgr 1 --proposal-structure 'DIRECT'    --tilt-structure 'NONE'   --log-group 'gdm-v3.0.5-fivo-sgr'             > /dev/null 2>&1 &
    python _test_fivo.py --seed $i --use-sgr 1 --proposal-structure 'DIRECT'    --tilt-structure 'DIRECT' --log-group 'gdm-v3.0.6-fivo-aux-sgr'         > /dev/null 2>&1 &
done

echo "All Done!"
