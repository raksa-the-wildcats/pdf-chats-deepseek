# Fix for ChromaDB SQLite compatibility on Streamlit Cloud
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from config import Config
from utils.qa_chain import QAChain

def initialize_session_state():
    """Initialize session state variables."""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'message_reasoning' not in st.session_state:
        st.session_state.message_reasoning = {}

def initialize_chatbot():
    """Initialize the chatbot and knowledge base."""
    try:
        Config.validate()
        st.session_state.qa_chain = QAChain()
        success = st.session_state.qa_chain.initialize_knowledge_base()
        if success:
            st.session_state.is_initialized = True
            return "✅ Chatbot initialized successfully!"
        else:
            return "❌ Failed to initialize knowledge base. Please check if PDFs are in the data/pdfs directory."
    except Exception as e:
        return f"❌ Initialization error: {str(e)}"

def refresh_knowledge_base():
    """Refresh the knowledge base."""
    if st.session_state.qa_chain:
        success = st.session_state.qa_chain.initialize_knowledge_base(force_rebuild=True)
        if success:
            return "✅ Knowledge base refreshed successfully!"
        else:
            return "❌ Failed to refresh knowledge base."
    return "❌ Chatbot not initialized."

def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []
    st.session_state.message_reasoning = {}

def is_greeting_or_casual(message):
    """Check if message is a greeting or casual input."""
    casual_patterns = [
        "hi", "hello", "hey", "hiya", "greetings",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye",
        "how are you", "what's up", "sup"
    ]
    message_lower = message.lower().strip()
    return any(pattern in message_lower for pattern in casual_patterns) or len(message.strip()) < 5

def get_casual_response(message):
    """Generate appropriate response for casual inputs."""
    message_lower = message.lower().strip()
    
    if any(greeting in message_lower for greeting in ["hi", "hello", "hey", "hiya", "greetings"]):
        return "Hello! I'm here to help with web accessibility questions. You can ask me about WCAG guidelines, ARIA labels, keyboard navigation, color contrast, or any other accessibility topics."
    
    elif any(thanks in message_lower for thanks in ["thanks", "thank you"]):
        return "You're welcome! Feel free to ask me any other accessibility questions."
    
    elif any(bye in message_lower for bye in ["bye", "goodbye"]):
        return "Goodbye! Come back anytime you have accessibility questions."
    
    elif any(how in message_lower for how in ["how are you", "what's up", "sup"]):
        return "I'm doing well and ready to help with accessibility questions! What would you like to know about web accessibility?"
    
    else:
        return "I'm here to help with web accessibility questions. Try asking about WCAG guidelines, ARIA labels, keyboard navigation, or color contrast!"

def get_example_questions():
    """Get example questions for the interface."""
    return [
        "What are the WCAG 2.1 guidelines for color contrast?",
        "How do I make images accessible?",
        "What is the proper way to use ARIA labels?",
        "How can I make forms more accessible?",
        "What are the requirements for keyboard navigation?"
    ]

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon="♿",
        layout="wide"
    )
    
    # Initialize session state FIRST
    initialize_session_state()
    
    # Header
    st.title(Config.APP_TITLE)
    st.markdown(Config.APP_DESCRIPTION)
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Controls")
        
        # Initialize button
        if st.button("Initialize Chatbot", type="primary"):
            with st.spinner("Initializing chatbot..."):
                result = initialize_chatbot()
                if "✅" in result:
                    st.success(result)
                else:
                    st.error(result)
        
        # Status indicator
        status_color = "🟢" if st.session_state.is_initialized else "🔴"
        status_text = "Ready" if st.session_state.is_initialized else "Not initialized"
        st.markdown(f"**Status:** {status_color} {status_text}")
        
        st.divider()
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                clear_chat()
                st.rerun()
        
        with col2:
            if st.button("Refresh KB"):
                with st.spinner("Refreshing knowledge base..."):
                    result = refresh_knowledge_base()
                    if "✅" in result:
                        st.success(result)
                    else:
                        st.error(result)
        
        st.divider()
        
        # Example questions
        st.header("💡 Quick Start")
        example_questions = get_example_questions()
        for i, question in enumerate(example_questions):
            # Create shorter, cleaner button text
            if "WCAG" in question:
                button_text = "Color Contrast Guidelines"
            elif "images" in question:
                button_text = "Image Accessibility"
            elif "ARIA" in question:
                button_text = "ARIA Labels Usage"
            elif "forms" in question:
                button_text = "Accessible Forms"
            elif "keyboard" in question:
                button_text = "Keyboard Navigation"
            else:
                button_text = question[:25] + "..."
            
            if st.button(button_text, key=f"example_{i}", use_container_width=True):
                st.session_state.example_question = question
                st.rerun()
    
    # Main chat interface
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                # Show reasoning toggle at the top of assistant responses that have reasoning
                if (message["role"] == "assistant" and 
                    hasattr(st.session_state, 'message_reasoning') and 
                    str(i) in st.session_state.message_reasoning):
                    with st.expander("🧠 Reasoning", expanded=False):
                        st.markdown(st.session_state.message_reasoning[str(i)])
                
                st.markdown(message["content"])
    
    # Chat input handling
    if not st.session_state.is_initialized:
        st.info("Please initialize the chatbot first using the sidebar.")
        # Still show input but disabled
        st.chat_input("Ask a question about web accessibility...", disabled=True)
    else:
        # Handle example question click
        if hasattr(st.session_state, 'example_question'):
            prompt = st.session_state.example_question
            del st.session_state.example_question
            
            # Process the example question immediately
            process_new_message(prompt)
        
        # Show chat input at the bottom
        prompt = st.chat_input("Ask a question about web accessibility...")
        if prompt:
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message right away
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and show assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    assistant_response = generate_response(prompt)
                    
                    # Show the complete response once generated
                    if assistant_response.get("reasoning"):
                        with st.expander("🧠 Reasoning", expanded=False):
                            st.markdown(assistant_response["reasoning"])
                    
                    st.markdown(assistant_response["content"])
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response["content"]})
            
            # Store reasoning if present
            if assistant_response.get("reasoning"):
                message_index = len(st.session_state.messages) - 1
                st.session_state.message_reasoning[str(message_index)] = assistant_response["reasoning"]
            
            st.rerun()

def generate_response(prompt):
    """Generate assistant response and return structured data."""
    try:
        # Check if it's a casual input/greeting
        if is_greeting_or_casual(prompt):
            return {
                "content": get_casual_response(prompt),
                "reasoning": None
            }
        else:
            # Get answer from QA chain for accessibility questions
            result_data = st.session_state.qa_chain.get_answer(prompt)
            
            # Parse DeepSeek response to separate thinking from answer
            full_response = result_data["answer"]
            thinking_content = ""
            main_answer = full_response
            
            # Check if response contains <think> tags
            if "<think>" in full_response and "</think>" in full_response:
                start_idx = full_response.find("<think>")
                end_idx = full_response.find("</think>") + len("</think>")
                thinking_content = full_response[start_idx:end_idx]
                main_answer = full_response[end_idx:].strip()
            
            # Format clean response
            response = main_answer
            if result_data["sources"]:
                response += f"\n\n**Sources:** {', '.join(result_data['sources'])}"
            
            # Prepare reasoning content
            reasoning = None
            if thinking_content:
                reasoning = thinking_content.replace("<think>", "").replace("</think>", "").strip()
            
            return {
                "content": response,
                "reasoning": reasoning
            }
        
    except Exception as e:
        return {
            "content": f"❌ Error processing your message: {str(e)}",
            "reasoning": None
        }

def process_new_message(prompt):
    """Process a new user message and generate response (for example questions)."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate assistant response
    assistant_response = generate_response(prompt)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response["content"]})
    
    # Store reasoning if present
    if assistant_response.get("reasoning"):
        message_index = len(st.session_state.messages) - 1
        st.session_state.message_reasoning[str(message_index)] = assistant_response["reasoning"]
    
    # Refresh the page to show new messages
    st.rerun()

if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Please create one with your REPLICATE_API_TOKEN")
        print("Example .env content:")
        print("REPLICATE_API_TOKEN=your_api_token_here")
    
    # Run the Streamlit app
    main()