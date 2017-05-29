#-*- coding: UTF-8 -*-
import os
def text_create(name, msg):
    desktop_path = 'F:\mission\\fenlei\positive\\'
    full_path = desktop_path + name + '.txt'
    file = open(full_path,'w',encoding="UTF-8")
    file.write(msg)
    file.close()
i = 20714



for line in open('F:\mission\\pos.txt',"r",encoding="UTF-8"):
    text_create(str(i),line)
    i+=1





