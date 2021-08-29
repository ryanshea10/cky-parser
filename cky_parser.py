# -*- coding: utf-8 -*-
"""
@author: ryan.shea
"""

import nltk
import fileinput
import tree
from collections import Counter
from math import log10

# functions to get grammar rules and counts from parse trees
def get_rules(tree):
    tokens = [i for i in tree.leaves()]
    rules_and_tokens = [i for i in tree.bottomup()]

    rules=[i for i in rules_and_tokens if i not in tokens]
    rulez=[get_children(i) for i in rules]
    return rulez


def get_children(node):
    lst=[node.label]+[x.label for x in node.children]
    return tuple(lst)


rules={}
for line in fileinput.input(files='train_trees_pre_unk.txt'):
    t=tree.Tree.from_str(line)
    current_rules=get_rules(t)
    current_counts=Counter(current_rules)
    for r in current_counts:
        rules[r]=rules.get(r,0)+current_counts[r]

# get list of top rules
top_rules = {}
for rule in rules:
    top_rules[rule[0]] = top_rules.get(rule[0], 0) + rules[rule]

# lambda function to find the log probability
get_log_prob = lambda r : log10(rules[r] / top_rules[r[0]])

# get grammar rule log probabilities
def make_grammar(rules):
    grammar={}
    for rule in rules:
        key=tuple(rule[1:])
        value=grammar.get(key, [])
        value.append(tuple([rule[0], get_log_prob(rule)]))
        grammar[key]=value
    return grammar

grammar=make_grammar(rules)




'CKY Algorithm'

# get final rule and log prob (rule at the leaf of the tree)
def get_rule(key, grammar, init=False):
    options=grammar.get((key,), grammar[('<unk>',)])
    back_options={new_key[0]:key for new_key in options}
    return options, back_options

# get the binary rules and probs to fill the parse table
def get_binary_rules(left_cell, down_cell, grammar, left_ind, down_ind):
    rules=[]
    childs=[]
    for child1 in left_cell:
        for child2 in down_cell:
            children=(child1[0], child2[0])
            if children not in grammar.keys():
                continue
            rule_candidates=grammar[children]
            for r in rule_candidates:
                rule=r[0]
                rule_prob=r[1]+child1[1]+child2[1]
                rules.append((rule, rule_prob))
                childs.append((rule,child1[0],child2[0], left_ind, down_ind))
    rules_to_convert={}
    childs_to_convert={}
    for i,j in zip(rules,childs):
        if i[0] in rules_to_convert.keys() and i[1]<rules_to_convert[i[0]]:
            continue
        rules_to_convert[i[0]]=i[1]
        childs_to_convert[j[0]]=(j[1],j[2],j[3],j[4])
        
    final_rules=[(key,value) for key,value in rules_to_convert.items()]
    
    return final_rules, childs_to_convert

# algorithm to fill out the parse table
def cky_parser(sentence, grammar):
    #sentence=nltk.word_tokenize(sent)
    n=len(sentence)+1
    table={}
    back={}
    for i in range(1,n):
        rule=get_rule(sentence[i-1], grammar, True)
        table[i-1, i]=rule[0]
        back[i-1, i]=rule[1]
    for l in range(2,n):
        for i in range(n-l):
            k=i+l
            for j in range(i+1, k):
                left_cell=table[i,j]
                down_cell=table[j,k]
                table[i,k]=table.get((i,k), [])+get_binary_rules(left_cell, down_cell, grammar, (i,j), (j,k))[0]
                r=get_binary_rules(left_cell, down_cell, grammar, (i,j), (j,k))[1]
                back[i,k]=back.get((i,k), {})
                back[(i,k)].update(r)
    return table,back

# Functions to backtrace through the parse table and output the highest 
# probability parse. The output format is the same as the treebank data

# The number of closing parentheses at the end of a word is equal to the 
# number of "down" branches directly above it.
# This function counts the number of down branches above a word and stores the
# number of parentheses to insert.
# There is a potential issue if there are two or more words below multiple down 
# branches, if this happens the dictionary will overwrite the number of parentheses
# to be inserted. This can be remedied by keying the dictionary on word position
# as opposed to word value
def backtrace_down(start_cell,back,down_count, close_branches):
    down_count+=1
    down_cell=back[start_cell[3]][start_cell[1]]
    if isinstance(down_cell,str):
        to_append=''.join([')']*down_count)
        close_branches[down_cell]=to_append
    else:
        backtrace_down(down_cell,back,down_count,close_branches)
    
    down_count=1
    left_cell=back[start_cell[2]][start_cell[0]]
    if isinstance(left_cell, str):
        return close_branches
    else:
        backtrace_down(left_cell,back,down_count,close_branches)
    return close_branches

# This function gets the proper word/grammar order for the parse
def backtrace_recur(start_cell,parse_order,back,branch_dict):          
    left_cell=back[start_cell[2]][start_cell[0]]
    if isinstance(left_cell,str):
        parse_order.append('('+start_cell[0])
        parse_order.append(left_cell+')')
    else:
        parse_order.append('('+start_cell[0])
        backtrace_recur(left_cell,parse_order,back,branch_dict) 
    
    down_cell=back[start_cell[3]][start_cell[1]]
    if isinstance(down_cell,str):    
        parse_order.append('('+start_cell[1])
        parse_order.append(down_cell+branch_dict[down_cell])
        return parse_order
    else:
        parse_order.append('('+start_cell[1])
        backtrace_recur(down_cell,parse_order,back,branch_dict)
        
# driver for the two functions above, backtrace recur gets the proper parse
# order and backtrace down inserts the correct closing parentheses
def backtrace(back, sent):   
    down_count=1
    parse_order=['(TOP']   
    start_cell=back[0,len(sent)]['TOP']
    
    left_cell=back[start_cell[2]][start_cell[0]]
    if isinstance(left_cell,str):
        parse_order.append('('+start_cell[0])
        parse_order.append(left_cell+')')
    else:
        parse_order.append('('+start_cell[0])
        branch_dict=backtrace_down(left_cell, back, down_count, {})
        backtrace_recur(left_cell,parse_order,back,branch_dict)        
       
    down_cell=back[start_cell[3]][start_cell[1]]
    if isinstance(down_cell,str):
        parse_order.append('('+start_cell[1])
        parse_order.append(down_cell+'))')
        return parse_order


# return the highest probability parse
def get_parse_and_prob(sent, grammar):
    sentence=nltk.word_tokenize(sent)
    cky_output=cky_parser(sentence, grammar)
    prob_dict=cky_output[0]
    back_dict=cky_output[1]
    if len(prob_dict[(0,len(sentence))])==0:
        return 'Cannot Parse'
    parse_prob=prob_dict[(0,len(sentence))][0][1]
    parse_list=backtrace(back_dict, sentence)
    parse_str=' '.join([i for i in parse_list])
    
    return parse_str,parse_prob



get_parse_and_prob('Thank you.', grammar)



