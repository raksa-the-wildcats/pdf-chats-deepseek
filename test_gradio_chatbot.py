import gradio as gr

def echo_chat(message, history):
    # Always return the history plus the new user and assistant messages
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"Echo: {message}"}
    ]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(label="Your Message")
    send = gr.Button("Send")

    send.click(echo_chat, inputs=[msg, chatbot], outputs=[chatbot])
    msg.submit(echo_chat, inputs=[msg, chatbot], outputs=[chatbot])

demo.launch() 