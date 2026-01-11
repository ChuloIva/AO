
### Option 3: Build a Simple "Clickable Text" Component (The Pro Solution)
If you want **exact control** (e.g., clicking a word toggles it efficiently without re-rendering the whole page heavily), the best option is a small **Custom Component**.

You don't need to build a full Python package. You can define a simple HTML/JS component directly in your script using `streamlit.components.v1.html`.

Here is a complete, working snippet you can copy-paste. It renders your tokens as a sentence. When you click a word, it sends the index back to Python.

```python
import streamlit as st
import streamlit.components.v1 as components

def clickable_token_sentence(tokens, active_indices=None):
    if active_indices is None:
        active_indices = []
    
    # CSS for the tokens
    style = """
    <style>
        .token {
            display: inline-block;
            margin: 2px;
            padding: 2px 6px;
            border-radius: 4px;
            cursor: pointer;
            font-family: sans-serif;
            transition: background 0.2s;
        }
        .token:hover {
            background-color: #e0e0e0;
        }
        .token.active {
            background-color: #ff4b4b;
            color: white;
        }
    </style>
    """

    # JS to handle clicks and send data back to Streamlit
    script = """
    <script>
        function sendIndex(index) {
            // Send the index back to Streamlit
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: index
            }, "*");
        }
    </script>
    """

    # Build HTML for tokens
    html_tokens = []
    for i, token in enumerate(tokens):
        is_active = "active" if i in active_indices else ""
        # We use an onclick handler to call the JS function
        html = f'<span class="token {is_active}" onclick="sendIndex({i})">{token}</span>'
        html_tokens.append(html)
    
    full_html = f"{style}{script}<div>{' '.join(html_tokens)}</div>"
    
    # Render the component
    # We use a static key to prevent full re-mounting, but the return value updates
    clicked_index = components.html(full_html, height=100, scrolling=True)
    
    return clicked_index

# --- Main App ---

if "selected_indices" not in st.session_state:
    st.session_state.selected_indices = set()

tokens = ["<|im_start|>", "system", "You", "are", "a", "doctor", "in", "an", "emergency", "room", "."]

# NOTE: standard components.html doesn't return values directly in the same way 
# fully custom components do. For a quick script-based solution, the best UX 
# is to use the 'st_click_detector' library if you don't want to compile React.

# RECOMMENDATION: Use st-click-detector for the easiest "HTML-like" interactivity
# pip install st-click-detector
try:
    from st_click_detector import click_detector
    
    st.markdown("### Click tokens to select:")
    
    html_content = ""
    for i, token in enumerate(tokens):
        color = "#ff4b4b" if i in st.session_state.selected_indices else "transparent"
        text_color = "white" if i in st.session_state.selected_indices else "inherit"
        # We wrap each token in an HTML link with a unique ID
        html_content += f'<a href="#" id="{i}" style="text-decoration:none; color:{text_color}; background-color:{color}; padding:4px; border-radius:4px; margin:2px;">{token}</a> '
    
    clicked_id = click_detector(html_content)
    
    if clicked_id:
        idx = int(clicked_id)
        if idx in st.session_state.selected_indices:
            st.session_state.selected_indices.remove(idx)
        else:
            st.session_state.selected_indices.add(idx)
        st.rerun()

except ImportError:
    st.error("Please install st-click-detector: `pip install st-click-detector`")

st.write("Selected Indices:", st.session_state.selected_indices)
```

### Recommendation
I strongly suggest **Option 3 using `st-click-detector`**.

It allows you to write standard HTML (so your tokens flow like a real sentence) but treats every `<a>` tag as a clickable event that returns the ID (index) to Python. It is much lighter than a full custom component and solves your "grid of buttons" problem immediately.