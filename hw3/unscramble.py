# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:44:15 2018

@author: sspie
"""

import numpy as np

def unscramble(mat, keys, ofstW, ofstH, bw, bh, W, H):
    
    col = 1
    row = 1
    blocks = []
    
    #for loop to gather blocks and put them in a list
    for i in range(0,len(keys)):
        c = col -1 
        r = row - 1
        
        col_str = ofstW + bw*c 
        col_stp = ofstW + bw*c+bw 
        row_str = ofstH + bh*r
        row_stp = ofstH + bh*r+bh
        block = np.array(mat[row_str:row_stp,col_str:col_stp], copy=True)
        blocks.append(block)
      
        if row == H:
            row = 0
            col = col + 1
            
        row = row + 1
    
    col = 1 #reset columns
    row = 1 #reset rows
    
    #second, similiar, for loop places blocks in there new spot!
    for i in range(0,len(keys)):
        #get fist block by keys index
        index = keys[i]-1
        block = blocks[index]
        
        c = col -1 
        r = row - 1
        
        col_str = ofstW + bw*c 
        col_stp = ofstW + bw*c+bw
        row_str = ofstH + bh*r
        row_stp = ofstH + bh*r+bh
        mat[row_str:row_stp,col_str:col_stp] = block #replace section of array
      
        
        if row == H:
            row = 0
            col = col + 1
            
        row = row + 1
    
    return mat
    
    
