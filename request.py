#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'T1':2, 'RH_1':9, 'T2':6, 'RH_2':2, 'T3':9, 'RH_3':6,'T4':2, 'T5':9, 'T6':6, 'RH_6':2, 'T7':9, 'RH_7':6, 'T8':2, 'RH_8':9, 'RH_9':6, 'T_out':2, 'Press_mm_hg':9, 'RH_out':6, 'Windspeed':2, 'hours':9})

print(r.json())

