#!/bin/bash

for name in BNZ IPH; do
    echo $name
    antechamber -i ./data/$name.sdf -fi mdl -o ./output/$name.mol2 -fo mol2 -c bcc
    parmchk2 -i ./output/$name.mol2 -f mol2 -o ./output/$name.frcmod
done
tleap -f ./script/make_prmtop.leaprc


