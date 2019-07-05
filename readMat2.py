# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:44 2019

@author: czpin
"""

"""
读取txt文件
该文本中的分割符既有空格又有制表符（\t），sep参数用\s+，可以匹配任何空格。
"""
#header=None:没有每列的column name，可以自己设定
#encoding='gb2312':其他编码中文显示错误
#sep=',':用逗号来分隔每行的数据
#index_col=0:设置第1列数据作为index
import pandas as pd
data_log2CNA = pd.read_csv('F:\Python_Project\lgg_tcga_pan_can_atlas_2018\data_log2CNA.txt', sep='\t' )
data_RNA_Seq_v2_mRNA_median_Zscores = pd.read_csv('F:\Python_Project\lgg_tcga_pan_can_atlas_2018\data_RNA_Seq_v2_mRNA_median_Zscores.txt', sep='\t', low_memory=False)
data_mutations_mskcc = pd.read_csv('F:\Python_Project\lgg_tcga_pan_can_atlas_2018\data_mutations_mskcc.txt', sep='\t', low_memory=False)
