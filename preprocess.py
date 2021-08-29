#!/usr/bin/env python

import fileinput
import tree



with open('train_trees_preprocessed.txt', 'w', encoding='utf8') as inp:
    for line in fileinput.input(files='train_trees.txt'):
        t = tree.Tree.from_str(line)
    
        # Binarize, inserting 'X*' nodes.
        t.binarize()
    
        # Remove unary nodes
        t.remove_unit()
    
        # The tree is now strictly binary branching, so that the CFG is in Chomsky normal form.
    
        # Make sure that all the roots still have the same label.
        assert t.root.label == 'TOP'
    
        inp.write(t.__str__())
        inp.write('\n')
    
    
