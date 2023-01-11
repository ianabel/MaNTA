#!/bin/bash

MTS=../solver


function stdout_test_case {
	local INPUT="$CASE.conf"
	local OUTPUT="$CASE.plot"
	local REF="$CASE.report"
	$MTS $INPUT > $OUTPUT 2>/dev/null;
	
	if diff -q $OUTPUT $REF >/dev/null;
	then
		rm -f $OUTPUT;
		return 0;
	else
		return 1;
	fi

}

function fileout_test_case {
	local INPUT="$CASE.conf"
	local OUTPUT="$CASE.plot"
	local REF="$CASE.report"
	$MTS $INPUT >/dev/null 2>/dev/null;
	
	if diff -q $OUTPUT $REF >/dev/null;
	then
		rm -f $OUTPUT;
		return 0;
	else
		return 1;
	fi
}

STDOUT_CASES=()
FILEOUT_CASES=(1DLinearTest 3VarLinearTest)

for CASE in ${STDOUT_CASES[@]}; do
	if stdout_test_case;
	then
		echo "Reference input $CASE.conf produces expected output";
	else
		echo "Reference input $CASE.conf failed to produce correct output; Failing output retained as $CASE.plot";
		exit 1;
	fi
done

for CASE in ${FILEOUT_CASES[@]}; do
	if fileout_test_case;
	then
		echo "Reference input $CASE.conf produces expected output";
	else
		echo "Reference input $CASE.conf failed to produce correct output; Failing output retained as $CASE.out";
		exit 1;
	fi
done
