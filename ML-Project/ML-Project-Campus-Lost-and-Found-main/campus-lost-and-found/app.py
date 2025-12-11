# File: app.py
# Purpose: Primary interface for the Campus Recovery System.
# Updated: Refactored for unique structure and optimized flow.

import streamlit as st
import os
import time
from modules import auth, db, features

# --- SYSTEM CONFIGURATION ---
IMG_STORAGE = "data/item_images"
if not os.path.exists(IMG_STORAGE):
    os.makedirs(IMG_STORAGE)

# distinct page title and layout
st.set_page_config(page_title="UniFind: Lost & Recovered", page_icon="🕵️", layout="wide")

# --- GLOBAL CONSTANTS ---
ITEM_CATEGORIES = [
    "Backpack", "Bracelet", "Calculator", "Charger", "Earphones", 
    "Headphones", "Keyboard", "Keys", "Laptop", "Mouse", 
    "Smartphone", "Waterbottle", "Wristwatch", "Other"
]

# --- STATE INITIALIZATION ---
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# ==========================
# UTILITY FUNCTIONS
# ==========================
def process_image_upload(file_obj):
    """Handles writing the uploaded image buffer to the local storage."""
    if not file_obj:
        return None
    try:
        # Create a unique filename using epoch time
        unique_name = f"{int(time.time())}_{file_obj.name}"
        full_path = os.path.join(IMG_STORAGE, unique_name)
        
        with open(full_path, "wb") as buffer:
            buffer.write(file_obj.getbuffer())
        
        return full_path
    except Exception as err:
        st.error(f"Upload failed: {err}")
        return None

# ==========================
# PAGE VIEWS
# ==========================
def view_auth():
    st.markdown("## 🔐 Access Portal")
    
    t_login, t_signup = st.tabs(["Sign In", "New Account"])
    
    # Login Logic
    with t_login:
        u_name = st.text_input("User ID")
        u_pass = st.text_input("Password", type="password")
        
        if st.button("Authenticate", key="btn_login"):
            user_data = auth.login_user(u_name, u_pass)
            if user_data:
                st.session_state['current_user'] = user_data
                st.toast(f"Hello, {user_data['username']}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Credentials not recognized.")

    # Registration Logic
    with t_signup:
        reg_user = st.text_input("Choose Username")
        reg_pass = st.text_input("Choose Password", type="password")
        reg_contact = st.text_input("Contact Info (Email/Phone)")
        
        if st.button("Register", key="btn_reg"):
            if not (reg_user and reg_pass and reg_contact):
                st.warning("All fields are mandatory.")
            else:
                uid = auth.register_user(reg_user, reg_pass, reg_contact)
                if uid:
                    st.success("Profile created! Switch to 'Sign In' tab.")
                else:
                    st.error("Username is already taken.")

def view_dashboard():
    active_user = st.session_state['current_user']
    
    # Sidebar Setup
    with st.sidebar:
        st.write(f"Logged in as: **{active_user['username']}**")
        if st.button("Sign Out"):
            st.session_state['current_user'] = None
            st.rerun()
        
        st.divider()
        nav_mode = st.radio("Menu", ["Post Item", "Find Match"])

    # Router
    if nav_mode == "Post Item":
        render_submission_form(active_user)
    elif nav_mode == "Find Match":
        render_search_engine(active_user)

def render_submission_form(user_obj):
    st.subheader("📝 Submit an Item Report")
    
    # State for auto-classification
    if 'suggested_idx' not in st.session_state:
        st.session_state['suggested_idx'] = 0 

    # Layout: Swapped columns for visual difference
    left_col, right_col = st.columns([1, 1])
    
    img_file = None
    
    with right_col:
        st.markdown("### 1. Image Upload")
        img_file = st.file_uploader("Upload photo for AI Analysis", type=["jpg", "png", "jpeg"])
        
        if img_file:
            st.image(img_file, width=250)
            # Perform AI Check immediately on upload
            saved_path = process_image_upload(img_file)
            ai_tag = features.predict_category(saved_path)
            
            if ai_tag:
                st.info(f"AI identified this as: **{ai_tag}**")
                # Auto-select the dropdown
                for i, cat in enumerate(ITEM_CATEGORIES):
                    if ai_tag.lower() == cat.lower():
                        st.session_state['suggested_idx'] = i
                        break

    with left_col:
        st.markdown("### 2. Details")
        rpt_type = st.radio("Status:", ["I LOST this", "I FOUND this"], horizontal=True)
        db_mode = "LOST" if "LOST" in rpt_type else "FOUND"
        
        # Dropdown defaults to AI suggestion
        sel_category = st.selectbox(
            "Item Category", 
            ITEM_CATEGORIES, 
            index=st.session_state['suggested_idx']
        )
        
        txt_desc = st.text_area("Detailed Description", placeholder="e.g. Red casing, scratch on the back...")

        if st.button("Save Report", type="primary"):
            if not txt_desc or not img_file:
                st.error("Image and description are required.")
            else:
                with st.spinner("Indexing features..."):
                    # Save and Process
                    final_path = process_image_upload(img_file)
                    v_vec = features.extract_visual_vector(final_path)
                    t_vec = features.extract_text_vector(txt_desc)
                    
                    if final_path and v_vec is not None:
                        db.add_item(
                            user_obj['id'], 
                            db_mode, 
                            sel_category, 
                            txt_desc, 
                            final_path, 
                            v_vec, 
                            t_vec
                        )
                        st.success("Report saved! System is looking for matches.")
                    else:
                        st.error("Processing failed. Try a different image.")

def render_search_engine(user_obj):
    st.subheader("🔍 Intelligent Search")
    
    # Search controls
    search_context = st.selectbox("I am looking for...", 
                                ["Items I lost (Search Found Db)", "Owners for item I found (Search Lost Db)"])
    
    target_db = "FOUND" if "lost" in search_context else "LOST"
    
    # Fetch DB Candidates
    db_items = db.get_candidates(target_db)
    
    if not db_items:
        st.warning(f"The '{target_db}' database is currently empty.")
        return

    # Search Inputs
    c1, c2 = st.columns(2)
    q_txt = c1.text_input("Keyword Search")
    q_img = c2.file_uploader("Visual Search (Image)", type=["jpg", "png", "jpeg"])
    
    run_search = st.button("Analyze Matches")

    if run_search:
        st.divider()
        scored_results = []
        
        # Vectorize queries
        q_txt_vec = features.extract_text_vector(q_txt) if q_txt else None
        q_vis_vec = None
        if q_img:
             path_temp = process_image_upload(q_img)
             q_vis_vec = features.extract_visual_vector(path_temp)

        # Scoring Logic
        for row in db_items:
            final_score = 0.0
            
            # Case 1: Hybrid
            if q_txt_vec is not None and q_vis_vec is not None:
                final_score = features.calculate_hybrid_score(
                    q_vis_vec, row['features_color'], 
                    q_txt_vec, row['features_text']
                )
            # Case 2: Text only
            elif q_txt_vec is not None:
                final_score = features.get_text_similarity(q_txt_vec, row['features_text'])
            # Case 3: Image only
            elif q_vis_vec is not None:
                final_score = features.get_visual_similarity(q_vis_vec, row['features_color'])
            
            # Filter low relevance
            if final_score > 0.01:
                scored_results.append((final_score, row))
        
        # Sort descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_results:
            st.caption("No relevant matches found.")
            return

        st.write(f"**{len(scored_results)}** Matches Detected:")
        
        for score, data in scored_results:
            # Determine visual indicator color
            indicator = "🟢" if score > 0.75 else "🟡" if score > 0.45 else "🔴"
            
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])
                
                with col_img:
                    if os.path.exists(data['image_path']):
                        st.image(data['image_path'], use_container_width=True)
                    else:
                        st.text("No Img")
                
                with col_info:
                    st.markdown(f"### {indicator} {data['category']}")
                    st.caption(f"Match Confidence: {score:.1%}")
                    st.write(f"**Details:** {data['description']}")
                    
                    # Explainability
                    if q_txt:
                        hits = features.explain_text_match(q_txt, data['description'])
                        if hits:
                            st.write(f"**Keywords:** {', '.join(hits)}")
                    
                    # Claim Button
                    btn_key = f"claim_btn_{data['id']}"
                    if st.button("Reveal Contact Info", key=btn_key):
                        st.success(f"📞 Contact: {data['contact_info']}")

# ==========================
# MAIN ENTRY POINT
# ==========================
def main():
    db.init_db()
    if st.session_state['current_user']:
        view_dashboard()
    else:
        view_auth()

if __name__ == "__main__":
    main()