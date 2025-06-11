shapes: leye reye mouth nose face rect poly
operations:
rotate shape deg -> rot shape
scale shape zoom -> zoomed shape
flip shape vert/horiz -> flipped shape
tile shape zoom -> ? terminal?
swap shape shape -> one of the shapes
copy shape shape -> dest shape
future operations:
fill shape rgb -> shape - set color in shape
invert shape -> shape - select all BUT the shape

TODO: targeting specific faces by index

meta:
clear -> clear all active manipulations
clear X -> clear manipulation with label X


# englishish
A(copy mouth to nose then rotate 45)
swap leye mouth

# clojure
(rotate (copy mouth nose) 45)
(swap leye mouth)

# something
A = copy(mouth, nose) -> rotate(45)
B = swap(leye, mouth)

# yaml
ops:
  - op: copy
	  src: mouth
		dest: nose
		then:
		  - op: rotate
			  deg: 45
		label: A
  - op: swap
	  a: leye
		b: mouth
		label: B

# haskell
A = rotate 45 copy mouth nose
B = swap leye mouth

parser
lang -> AST

interpreter
AST -> OperationTree

OR

parsterpreter
lang -> OperationTree

Thoughts...

No yaml. Prevents "realtime" capability of accpeting commands from stdin
