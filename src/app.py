import streamlit as st
from PIL import Image




st.title("📝 Visual Question Answering Project", )

width_head = """
    <style>
        [data-testid="stApp"] {
                background: url(https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg);
                background-attachment: fixed;
                background-repeat: no-repeat;
                background-size: cover;
        }
        [data-testid="StyledLinkIconContainer" span] {
            white-space : nowrap;
            background: -webkit-linear-gradient(#eee, #333);
            -webkit-background-clip: text;
            # -webkit-text-fill-color: transparent;
        }   
        .st-emotion-cache-zt5igj span {
            white-space: nowrap;
            background: linear-gradient(172deg, #c42525cf, #2130ae);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        [data-testid="element-container"]{
                text-align: -webkit-center;

        }
        [data-testid="baseButton-secondaryFormSubmit"]{
            padding: 12px 40px;
        }
    </style>
"""
st.markdown(width_head, unsafe_allow_html=True)
# Custom CSS to create a border
border_css = """
<style>
    .custom-border {
        border: 10px solid #4CAF50;  /* Green border */
        padding: 10px;
        border-radius: 5px;
    }
</style>
"""

st.markdown('<div class="custom-border">', unsafe_allow_html=True)
# st.write("This is inside a bordered box")
st1, st2 = st.columns(2)
image = None
if image == None:
    uploaded_file = st.file_uploader("Upload an article", type=("png", "jpg"))
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        
quesiton: str = ""
if image != None:
        with st.form("my_form"):
            col1, col2 = st.columns(2)
            with col1:
                question = st.text_area("Enter text:", placeholder="What is your question?")
            with col2:
                st.image(image, caption="Uploaded Image.", use_column_width=True)
            submitted = st.form_submit_button("Submit")     

st.markdown('</div>', unsafe_allow_html=True)


    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    # elif submitted:
    #     generate_response(text)


# if uploaded_file and question and not anthropic_api_key:
#     st.info("Please add your Anthropic API key to continue.")

# if uploaded_file and question and anthropic_api_key:
#     article = uploaded_file.read().decode()
#     prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n
#     {article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""

#     client = anthropic.Client(api_key=anthropic_api_key)
#     response = client.completions.create(
#         prompt=prompt,
#         stop_sequences=[anthropic.HUMAN_PROMPT],
#         model="claude-v1", #"claude-2" for Claude 2 model
#         max_tokens_to_sample=100,
#     )
#     st.write("### Answer")
#     st.write(response.completion)