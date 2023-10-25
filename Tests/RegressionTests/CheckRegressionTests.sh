#!/bin/bash

MANTA=../../$SOLVER

echo "Testing Reference Runs"

function fileout_test_case {
	local INPUT="$CASE.conf"
	local OUTPUT="$CASE.dat"
	local NC_OUTPUT="$CASE.nc"
	local REF="$CASE.ref.dat"
	$MANTA $INPUT >/dev/null 2>/dev/null;
	
	if diff -q $OUTPUT $REF >/dev/null;
	then
		rm -f $OUTPUT $NC_OUTPUT;
		return 0;
	else
		return 1;
	fi
}

STDOUT_CASES=()
FILEOUT_CASES=(LinearDiffusion MatTest MatTestAlpha)

for CASE in ${FILEOUT_CASES[@]}; do
	if fileout_test_case;
	then
		echo "Reference input $CASE.conf produces expected output";
	else
		echo "Reference input $CASE.conf failed to produce correct output; Failing output retained as $CASE.dat";
		exit 1;
	fi
done
