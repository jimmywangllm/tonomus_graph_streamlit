##########tonomus_graph_streamlit.py###########

'''

streamlit run tonomus_graph_streamlit.py


'''


import streamlit
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np 
import os



import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk


nodes = []
edges = []

entity_color = pd.read_excel('entity_for_graph.xlsx')
entity_color = entity_color.to_dict('records')

triplets = pd.read_excel('triplets.xlsx')
triplets = triplets.to_dict('records')



config = Config(
directed=True, 
physics=True, 
hierarchical=True,
#width=3000,
#height=1600,
# **kwargs
)




if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Show me the knowledge graph of Aramus and Safana."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    entities = []

    for r in triplets:
        if r['subject'].lower() in prompt.lower() or r['object'].lower() in prompt.lower():
            edges.append( 
                Edge(
                source=r["subject"], 
                label=r["relation"], 
                target=r["object"], 
                # **kwargs
                ) 
                ) 
            entities.append(r['subject'].lower())
            entities.append(r['object'].lower())

    for r in entity_color:
        if r['entity'].lower() in entities:
            nodes.append( 
                Node(
                id= r['entity'], 
                label= r['entity'][0:16], 
                title= r['entity'], 
                size=25, 
                font={'color': 'white'},
                color = r['color'],
                shape="circularImage",
                image=f"{r['entity']}.png",
                ))



    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        return_value = agraph(
            nodes=nodes, 
            edges=edges, 
            config=config,
            )
##########tonomus_graph_streamlit.py###########