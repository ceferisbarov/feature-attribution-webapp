import streamlit as st
from engine import explain, get_red_code

def highlight(prompt):
    result = explain(prompt)
    # # st.write("The current movie title is", result)
    span_template = '<span style="color: COLOR;">TEXT</span>'
    html_content = '<p>\n'
    for word, intensity in result:
        hex_intensity = format(intensity, '02x')
        # Create the color code
        color = f"#{hex_intensity}0000"
        html_content += span_template.replace("COLOR", color).replace("TEXT", word + " ")
        html_content += "\n"
    html_content += "<p>\n"

    return html_content

add_selectbox = st.sidebar.selectbox(
    "Select an LLM.",
    ("GPT-3.5", "LLaMA 2 7B", "Mistral 7B")
)

st.logo("assets/horizontal.png", icon_image="assets/icon.png")

prompt = st.text_input("Enter a prompt:")
with st.empty():
    html_content = highlight(prompt)
    print("========================================")
    print(html_content)
    print("========================================")

    st.write(html_content)
    st.html(html_content)
