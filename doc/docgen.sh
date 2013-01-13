#!/bin/sh

# Automatic documentation generator
# Assumes that Discount has been installed and is in the path.
# Discount can be obtained here:
#   http://www.pell.portland.or.us/~orc/Code/discount/
#
# We assume all relevant documentation is of the following format.
# The first line of a documentation block starts with at least 8
# forward slash characters starting from the first column.
# Each subsequent line begins with two forward slashes starting
# from the first column.
# The prototype that follows is extracted extract all lines up to the
# first open curly brace.

RNP_ROOT=..

for file in `cat docgen_list.txt`; do
	#docgen_file.sh $file
	echo $file
	BSNM=`basename $file | sed 's/\(.*\)\..*/\1/'`
	# This top script is the main extraction of comment blocks
	awk '# s is the state (1 when we are in comment block, 2 in prototype)
		BEGIN                { s=0; b=0      } # Initialize state
		 /^\/\/\/\/\/\/\/\// { s=1; print "***"; next } # Encountered 8 slashes, print blank line instead
		!/^\/\//             { if(s==1) s=2 } # Stopped encountering 2 slashes
		 /\{/                { b=1;          }
		!/\{/                { b=0;          }
		{
			if(s == 1){ print }
			else if(s == 2){
				if($0 != "") print "\t" $0
			}
			if((s == 2) && (b == 1)){ s=0 }
		}
		' $RNP_ROOT/$file |
	sed -e 's/{$//' | # strip off trailing open brace
	sed -e 's/^\/\/ \?//' |
	awk '# Format argument lists (s=2 within argument list)
		BEGIN{ s=0 }
		 /^Arguments/ { s=1 }
		 /^[a-zA-Z]/  { w=1 }
		!/^[a-zA-z]/  { w=0 }
		 /^\t/        { if(t==0){ print "### Prototype" } s=0; t=1 }
		!/^\t/        { t=0; }
		{
			if(s == 1){
				print "### " $0
				s=2
			}else if(s == 2){
				if(w){
					print "=" $1 "="
					$1 = "    "
				}
				print
			}else{
				print
			}
		}
		' > gen/$BSNM.txt
	# Extract the title
	TITLE=`grep -B1 "^==" gen/$BSNM.txt | head -n 1`
	sed -e "s/{TITLE}/$TITLE/" docgen_header.txt > gen/$BSNM.html
	markdown -f +toc -T gen/$BSNM.txt >> gen/$BSNM.html
	cat docgen_footer.txt >> gen/$BSNM.html
done
