import streamlit as st

# Your Streamlit code
st.title("My Streamlit App")
st.write("Hello, this is running with uv!")

if __name__ == "__main__":
    # Workaround to launch streamlit app when run as a normal script
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, "", [], {})
